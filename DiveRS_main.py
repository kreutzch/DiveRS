import collections
from typing import List, Dict, Any, Tuple, OrderedDict

import numpy as np
import random
import math
import itertools
import DiveRS_sub
from DiveRS_sub import DiveRSSub

# thanks to Stelmakh et al. for providing the basic structure of the code: https://www.cs.cmu.edu/~istelmak/


# identifies papers to drop while trying to identify problematic papers for which new reviewers should be included
# seed - seed value for rng
# paper_list - list of papers from which ones to drop will be chosen
def find_papers_to_drop(seed, paper_list):
    p_ids = paper_list
    random.Random(seed).shuffle(p_ids)

    # drop half of the papers, at least one paper
    num_of_papers = math.ceil(len(paper_list) * 0.5)
    if num_of_papers == 0:
        return []

    dropped_papers = p_ids[0: num_of_papers]
    return dropped_papers


class DiveRSMain:
    # tolerance for integrality check
    _EPS = 1e-3

    # initialize the parameters
    ####################################################################################################################
    # demand - lambda, requested number of reviewers per paper
    # similarity - similarity matrix between reviewers and papers with COIs coded into as -1
    # ability - the maximum number of papers a reviewer can review
    # lowest_load - the minimum number of papers a reviewer needs to review
    # dependant_revs - pairs of reviewers which are not independent from each other
    # aca_ind - dict of reviewer to information whether this person works in industry or academia or both
    #           aca_ind[reviewer] : 2 = works in both, 1 = works in aca, 0 = works in ind
    # location - dict of reviewer to list of continents, in which they work
    #           location[reviewer] contains 0 = 'SA', 1 = 'AF', 2 = 'AN', 3 = 'AS', 4 = 'OC', 5 = 'NA', 6 = 'EU'
    # seniority - dict of reviewer to their seniority
    #           seniority[reviewer] : 0 = senior, 1 = advanced, 2 = junior
    # tries - number of tries to find most diverse assignment in main routine
    # similarity_erc
    # ability_erc
    # lowest_load_erc
    # dependant_revs_erc
    # aca_ind_erc
    # location_erc
    # seniority_erc
    # theta - similarity threshold under which reviewers are not considered to be assigned for a submission
    # kappa - max number of reviewers being included in one iteration of the main routine
    # function - transformation function for similarities
    # delta - accepted margin of error for max-flow calculation due to floating point addition
    # percentage - percentage of papers/reviewers to be dropped in each run while searching for problematic papers and
    #           later on when the PC has been extended when searching for the best assignment
    def __init__(self, demand, similarity, ability, lowest_load, dependant_revs, aca_ind, location, seniority, tries,
                 similarity_erc, ability_erc, lowest_load_erc, dependant_revs_erc, aca_ind_erc, location_erc,
                 seniority_erc, rev_id_to_key, rev_id_to_key_erc, theta, kappa, function=lambda x: x, delta=1e-5,
                 percentage=0.1):

        self.comb_cand = {}
        self.sim_threshold = theta
        self.similarity = similarity
        self.original_similarity = np.copy(self.similarity)
        self.num_rev = int(similarity.shape[0])
        self.num_rev_erc = int(similarity_erc.shape[0])
        self.num_papers = int(similarity.shape[1])
        self.rev_id_to_key = rev_id_to_key
        self.rev_id_to_key_erc = rev_id_to_key_erc
        self.tries = tries
        self.kappa = kappa
        self.problematic_papers_from_last_try = []

        if demand > self.num_rev:
            raise ValueError('Number of reviewers needs to be greater or equal to demand or reviewers for each paper.')

        a: List[int] = []
        if isinstance(ability, int):
            if ability * self.num_rev < self.num_papers * demand:
                raise ValueError('Ability of all reviewers * number of papers cannot be less than number of papers '
                                 '* demanded reviewers.')
            for r in range(self.num_rev):
                a.append(ability)

            self.ability = a
        else:
            self.ability = ability

        l: List[int] = []
        if isinstance(lowest_load, int):
            if lowest_load * self.num_rev > self.num_papers * demand:
                raise ValueError('Lowest load of all reviewers * number of papers cannot be greater than number of '
                                 'papers * demanded reviewers.')

            for r in range(self.num_rev):
                l.append(lowest_load)

            self.lowest_load = l
        else:
            self.lowest_load = lowest_load

        a_erc: List[int] = []
        if isinstance(ability_erc, int):
            for r in range(self.num_rev_erc):
                a_erc.append(ability_erc)

            self.ability_erc = a_erc
        else:
            self.ability_erc = ability_erc

        l_erc: List[int] = []
        if isinstance(lowest_load_erc, int):
            for r in range(self.num_rev_erc):
                l_erc.append(lowest_load_erc)

            self.lowest_load_erc = l_erc
        else:
            self.lowest_load_erc = lowest_load_erc

        self.demand = int(demand)
        if self.demand < 2:
            raise ValueError('Number of reviewers per paper should be at least 2 otherwise location constraint cannot '
                             'be fulfilled.')

        self.function = function

        self.dependant_revs = dependant_revs
        self.aca_ind = aca_ind
        self.location_ids = {'SA': 0, 'AF': 1, 'AN': 2, 'AS': 3, 'OC': 4, 'NA': 5, 'EU': 6}
        self.location = location
        self.seniority = seniority

        self.similarity_erc = similarity_erc
        self.dependant_revs_erc = dependant_revs_erc
        self.aca_ind_erc = aca_ind_erc
        self.location_erc = location_erc
        self.seniority_erc = seniority_erc

        self.delta = delta
        self.percentage = percentage

        self.sim_threshold = theta

        # keeping track of reviewers and submissions which are out of scope
        self.ignored_reviewers = []
        self.ignored_submissions = []

        self.f = open('output.txt', 'w')

        self.ignored_prob_papers = []

    # delete dropped edges and ones with conflicts of interests from network
    # pairs - all pairs of reviewers and papers as edges
    # dropped_edges - list of reviewer-paper edges to exclude in the network
    def _cleanse_pairs(self, pairs, dropped_edges):
        pairs_wo_dropped_edges = []

        for curr_pair in pairs:
            if curr_pair not in dropped_edges and self.similarity[curr_pair[0], curr_pair[1]] > -1:
                pairs_wo_dropped_edges.append(curr_pair)

        return pairs_wo_dropped_edges

    # identifies edges corresponding to reviewer-paper decision pairs to drop
    # pairs - all pairs of reviewers and papers as edges
    # seed - seed value for rng
    def _find_edges_to_drop(self, pairs, seed):
        # drop reviewer-paper decision pairs
        num_of_pairs = int(len(pairs) * self.percentage)
        if num_of_pairs == 0:
            num_of_pairs = 1

        random.Random(seed).shuffle(pairs)
        return pairs[0: num_of_pairs]

    # compute fairness of a specific assignment
    # assignment - assignment of reviewers to all papers for which fairness will be computed
    def fairness(self, assignment):
        final_fairness = np.inf

        for paper in assignment:
            curr_fairness = sum([self.function(self.similarity[reviewer, paper]) for reviewer in assignment[paper]])
            if final_fairness > curr_fairness:
                final_fairness = curr_fairness
        return final_fairness

    # preparation of reviewer assignment step, handles identification of the best assignment
    # tries_upper_bound - int, how many runs of the specific type are conducted
    # drop_edges - boolean, indicates if assignment includes dropping of edges (happens in actual runs to help identify
    # the best assignment)
    def _assignment(self, tries_upper_bound, drop_edges=True):
        self.best_fairness = 0
        self.best_diversity = 0

        pairs = [[reviewer, paper] for (reviewer, paper) in
                 itertools.product(range(self.num_rev), range(self.num_papers))]

        # all candidate assignments
        self.assignments_from_all_runs = []
        self.fairness_from_all_runs = []

        try_diversity = -1

        subroutine_assignment: DiveRSSub = DiveRS_sub.DiveRSSub(self.similarity, self.dependant_revs, self.aca_ind,
                                                                self.location, self.seniority, [], self.demand,
                                                                self.ability, self.lowest_load, self.function)
        subroutine_assignment.initialize_model()

        for tries in range(0, tries_upper_bound):
            # drop predefined percentage of reviewer - submission relations
            if drop_edges:
                # compute assignment with dropping of edges
                assignment, found = subroutine_assignment.assignment_step_subroutine(dropped_edges=self.
                                                                                     _find_edges_to_drop(pairs, tries))

            else:
                # compute initial assignment without dropping of edges
                assignment, found = subroutine_assignment.assignment_step_subroutine()
            if found:
                curr_fairness = self.fairness(assignment)

                self.assignments_from_all_runs.append(assignment)
                self.fairness_from_all_runs.append(curr_fairness)

                # best candidate assignment
                if curr_fairness > self.best_fairness or self.best_fairness == 0:
                    self.fair_assignment = assignment
                    self.best_fairness = curr_fairness
                    self.fairness(self.fair_assignment)

                # compute assignment with highest diversity score
                curr_diversity = self.calculate_diversity_score(assignment)

                if curr_diversity > self.best_diversity or self.best_diversity == 0:
                    try_diversity = tries
                    self.diverse_assignment = assignment
                    self.best_diversity = curr_diversity

        if try_diversity > -1:
            self.f.write('suitable assignment computed \n')
            self.f.flush()

        return try_diversity > -1

    # check if current PC fulfils requirements to theoretically assign a senior reviewer to all manuscripts, this
    # disregards possible COIs
    def check_seniority(self):
        true_abilities = []

        for r in range(self.num_rev):
            if self.seniority[r] == 0:
                true_abilities.append(self.ability[r])

        if sum(true_abilities) < self.num_papers:
            return False, true_abilities

        return True, true_abilities

    # check if current PC fulfils requirements to theoretically assign an academia and non-academia (industry) reviewer
    # to all manuscripts, this disregards possible COIs
    def check_professional_background(self):
        true_abilities = [[], [], []]

        for r in range(self.num_rev):
            true_abilities[self.aca_ind[r]].append(self.ability[r])

        if sum(true_abilities[0]) + sum(true_abilities[2]) < self.num_papers or \
                sum(true_abilities[1]) + sum(true_abilities[2]) < self.num_papers:
            return False, true_abilities

        return True, true_abilities

    # called after new reviewer is inserted in PC to update theoretical capacities of PC pool
    # index - index of newly introduced PC member
    # bg_a - theoretical capacities of PC members in professional background
    # bg_l - theoretical capacities of PC members in location
    # bg_s - theoretical capacities of PC members in seniority
    def update_bg_variables(self, index, bg_a, bg_l, bg_s):
        # professional background
        if self.aca_ind_erc[index] == 2:
            bg_a[0] += 1
            bg_a[1] += 1
        bg_a[self.aca_ind_erc[index]] += 1

        # location
        for c_l in self.location_erc[index]:
            bg_l[self.location_ids[c_l]] += 1

        # seniority
        bg_s[self.seniority_erc[index]] += 1
        return bg_a, bg_l, bg_s

    # includes new reviewer candidate from ERC in PC
    # index - index of reviewer in ERC
    def include_new_reviewer(self, index):
        self.f.write('included ' + self.rev_id_to_key_erc[index] + '\n')
        self.f.flush()

        self.rev_id_to_key.append(self.rev_id_to_key_erc[index])

        self.similarity = np.append(self.similarity, np.reshape(self.similarity_erc[index], (1, -1)), axis=0)

        if self.sim_threshold > 0:
            for submission in range(self.num_papers):
                if self.similarity.item(len(self.aca_ind), submission) < self.sim_threshold:
                    self.similarity[len(self.aca_ind), submission] = -1

        self.aca_ind.append(self.aca_ind_erc[index])
        self.location.append(self.location_erc[index])
        self.seniority.append(self.seniority_erc[index])

        self.ability.append(self.ability_erc[index])
        self.lowest_load.append(self.lowest_load_erc[index])

    # adds new reviewers from ERC in PC based on missing seniority aspects
    # true_abilities - theoretical number of submissions which could be fulfilled with current PC
    # bg_a - theoretical capacities of PC members in professional background
    # bg_l - theoretical capacities of PC members in location
    # bg_s - theoretical capacities of PC members in seniority
    def add_new_reviewers_based_on_seniority(self, true_abilities, bg_a, bg_l, bg_s):
        # rank additional reviewer candidates descending by their average similarity to all submissions
        sum_sim_per_rev = {}

        for r in range(self.similarity_erc.shape[0]):
            if self.sim_threshold > 0:
                sum_sim_per_rev[r] = np.sum(self.similarity_erc[np.where(self.similarity_erc[r] >= self.sim_threshold)])
            else:
                sum_sim_per_rev[r] = np.sum(self.similarity_erc[np.where(self.similarity_erc[r] >= 0)])

        sum_sim_per_rev = sorted(sum_sim_per_rev.items(), key=lambda x: x[1], reverse=True)

        while sum(true_abilities) < self.num_papers:
            found = False

            # find one new reviewer to include
            ct = 0
            r: Tuple[int, float]
            for r in sum_sim_per_rev:
                # if reviewer is from desired group include them in PC
                if self.seniority_erc[r[0]] == 0:
                    # delete reviewer from candidates, include their ability in true abilities
                    del sum_sim_per_rev[ct]
                    true_abilities.append(self.ability_erc[r[0]])
                    # include reviewer in PC
                    self.include_new_reviewer(r[0])

                    # update bg variables
                    bg_a, bg_l, bg_s = self.update_bg_variables(r[0], bg_a, bg_l, bg_s)

                    # delete reviewer from ERC
                    self.drop_reviewer_from_erc(r[0], len(self.seniority) - 1)
                    self.num_rev += 1

                    found = True
                    break
                ct += 1

            # if no reviewer was found to include to satisfy the missing seniority
            if not found:
                raise ValueError('No new reviewer found so satisfy the missing seniority. Assignment not possible.')

        return bg_a, bg_l, bg_s

    # adds new reviewers from ERC in PC based on missing professional background aspects
    # true_abilities - theoretical number of submissions which could be fulfilled with current PC
    # bg_a - theoretical capacities of PC members in professional background
    # bg_l - theoretical capacities of PC members in location
    # bg_s - theoretical capacities of PC members in seniority
    def add_new_reviewers_based_on_professional_background(self, true_abilities, bg_a, bg_l, bg_s):
        # rank additional reviewer candidates descending by their average similarity to all submissions
        sum_sim_per_rev = {}

        for r in range(self.similarity_erc.shape[0]):
            sum_sim_per_rev[r] = np.sum(self.similarity_erc[r])

        sum_sim_per_rev = sorted(sum_sim_per_rev.items(), key=lambda x: x[1], reverse=True)

        while sum(true_abilities[0]) + sum(true_abilities[2]) < self.num_papers or \
                sum(true_abilities[1]) + sum(true_abilities[2]) < self.num_papers:
            reviewers_needed = 0

            if sum(true_abilities[0]) > sum(true_abilities[1]):
                reviewers_needed = 1
            elif sum(true_abilities[0]) < sum(true_abilities[1]):
                reviewers_needed = 0

            found = False

            # find one new reviewer to include
            r: Tuple[int, float]
            for r in sum_sim_per_rev:
                # if reviewer is from desired group include them in PC
                if self.aca_ind_erc[r[0]] == reviewers_needed or self.aca_ind_erc[r[0]] == 2:
                    # delete reviewer from candidates, include their ability in true abilities
                    del sum_sim_per_rev[r[0]]
                    true_abilities[self.aca_ind_erc[r[0]]].append(self.ability_erc[r[0]])
                    # include reviewer in PC
                    self.include_new_reviewer(r[0])

                    # update bg variables
                    bg_a, bg_l, bg_s = self.update_bg_variables(r[0], bg_a, bg_l, bg_s)

                    # delete reviewer from ERC
                    self.drop_reviewer_from_erc(r[0], len(self.aca_ind) - 1)
                    self.num_rev += 1

                    found = True
                    break

            # if no reviewer was found to include to satisfy the missing background
            if not found:
                raise ValueError('No new reviewer found so satisfy the missing professional background. '
                                 'Assignment not possible.')

        return bg_a, bg_l, bg_s

    # drops reviewers from ERC which are included in PC
    # index_old - old index of reviewer in ERC
    # index_new - new index of reviewer in PC
    def drop_reviewer_from_erc(self, index_old, index_new):
        # insert dependencies of new reviewer
        for pc_d in self.dependant_revs_erc[index_old]['pc']:
            self.dependant_revs.add((pc_d, index_new))
            self.dependant_revs.add((index_new, pc_d))

        # modify dependencies of all other ERC reviewers if they contain the new reviewer
        for r in range(len(self.dependant_revs_erc)):
            if r != index_old:
                new_erc = []
                for r_d in self.dependant_revs_erc[r]['ERC']:
                    if r_d < index_old:
                        new_erc.append(r_d)
                    elif r_d > index_old:
                        new_erc.append(r_d - 1)
                    elif r_d == index_old:
                        self.dependant_revs_erc[r]['pc'].append(index_new)

                self.dependant_revs_erc[r]['ERC'] = new_erc

        del self.dependant_revs_erc[index_old]

        # update id to key
        del self.rev_id_to_key_erc[index_old]

        self.similarity_erc = np.delete(self.similarity_erc, index_old, 0)

        del self.aca_ind_erc[index_old]
        del self.location_erc[index_old]
        del self.seniority_erc[index_old]

        del self.ability_erc[index_old]
        del self.lowest_load_erc[index_old]

    # set similarities between submissions and reviewers to -1 if they lie under a similarity threshold
    def calculate_thresholded_similarities(self):
        papers_to_drop = set()
        thresholded_similarities = self.similarity

        ct_reg = {}
        for submission in range(self.num_papers):
            ct_reg[submission] = 0

            curr_slice = self.similarity[:, submission]
            for var in range(len(curr_slice)):
                if curr_slice[var] < self.sim_threshold:
                    thresholded_similarities[var, submission] = -1
                else:
                    ct_reg[submission] = ct_reg[submission] + 1

        self.similarity = thresholded_similarities
        ct_erc = {}

        thresholded_similarities = self.similarity_erc
        for submission in range(self.num_papers):
            ct_erc[submission] = 0

            curr_slice = self.similarity_erc[:, submission]
            for var in range(len(curr_slice)):
                if curr_slice[var] < self.sim_threshold:
                    thresholded_similarities[var, submission] = -1
                else:
                    ct_erc[submission] = ct_erc[submission] + 1

            self.comb_cand[submission] = ct_reg[submission] + ct_erc[submission]
            if self.comb_cand[submission] < self.demand:
                papers_to_drop.add(submission)

        self.similarity_erc = thresholded_similarities

        self.delete_oos_papers(list(papers_to_drop), [])

    # insert as many new reviewers as needed for definitely problematic papers
    # def_problematic_papers - papers which do not have enough reviewers in PC without conflict of interest
    # prob_problematic_papers - papers which had high probability of being part of runs where no assignment was found
    # needed_rev_num_per_prob_sub - for all manuscripts in def_problematic_papers: demand - number of reviewers which
    # could be assigned to a specific manuscript
    # bg_a - theoretical capacities of PC members in professional background
    # bg_l - theoretical capacities of PC members in location
    # bg_s - theoretical capacities of PC members in seniority
    def insert_for_def_problematic_papers(self, def_problematic_papers, prob_problematic_papers,
                                          needed_rev_num_per_prob_sub, bg_a, bg_l, bg_s):
        # reviewer -> list of submissions in which they are needed
        def_new_rev_candidates = {}
        oos_papers = []

        for pp in def_problematic_papers:
            ct = int(needed_rev_num_per_prob_sub[pp])

            for new_rev in range(len(self.similarity_erc)):
                if ct == 0:
                    break
                # if new candidate has no conflict of interest, they can be looked at
                if self.similarity_erc.item(new_rev, pp) != -1:
                    cond_1 = self.sim_threshold > 0
                    cond_2 = cond_1 and self.similarity_erc.item(new_rev, pp) >= self.sim_threshold
                    if not cond_1 or cond_2:
                        ct = ct - 1

                        if new_rev not in def_new_rev_candidates:
                            def_new_rev_candidates[new_rev] = [0] * self.num_papers
                        def_new_rev_candidates[new_rev][pp] = 1

            if ct > 0:
                # found paper which is most likely out of scope of conference
                oos_papers.append(pp)

        # delete out of scope papers
        if oos_papers:
            prob_problematic_papers = self.delete_oos_papers(oos_papers, prob_problematic_papers)

        inserted_candidates: Dict[int, int] = {}
        # reviewers for submissions which definitely require new reviewers
        for candidate_id in def_new_rev_candidates:
            inserted_candidates[candidate_id] = len(self.aca_ind)

            self.include_new_reviewer(candidate_id)

            # update background variables
            bg_a, bg_l, bg_s = self.update_bg_variables(candidate_id, bg_a, bg_l, bg_s)

        # delete new reviewers from ERC
        inserted_candidates: OrderedDict[int, int] = collections.OrderedDict(sorted(inserted_candidates.items(),
                                                                                    reverse=True))
        for i_c in inserted_candidates:
            self.drop_reviewer_from_erc(i_c, inserted_candidates[i_c])

        self.num_rev += len(inserted_candidates)

        return bg_a, bg_l, bg_s, prob_problematic_papers

    # deletes papers from currently considered manuscripts which are considered out of scope
    # oos_papers - manuscripts which are considered to be out of scope of conference, not enough reviewers were found
    # for them such that a reviewer assignment could be conducted
    # prob_problematic_papers - submissions for which new reviewers might have to be included such that a reviewer
    # assignment can be constructed
    def delete_oos_papers(self, oos_papers, prob_problematic_papers):
        oos_papers.sort(reverse=True)

        for submission in oos_papers:
            if prob_problematic_papers:
                for sub in range(submission + 1, self.num_papers):
                    if sub in prob_problematic_papers:
                        swapped_paper = prob_problematic_papers[sub]
                    else:
                        swapped_paper = 1
                    prob_problematic_papers[sub - 1] = swapped_paper
                    if sub in prob_problematic_papers:
                        del prob_problematic_papers[sub]

            self.ignored_submissions.append(submission)

            self.similarity = np.delete(self.similarity, submission, 1)
            self.similarity_erc = np.delete(self.similarity_erc, submission, 1)

            self.num_papers -= 1
        return prob_problematic_papers

    # test if every reviewer has at least lower bound papers where similarity > similarity threshold and delete
    # reviewer otherwise
    def test_reviewer_loads(self):
        rev_to_ignore = []
        for reviewer in range(self.num_rev):
            ct = 0
            for submission in range(self.num_papers):
                if self.similarity.item(reviewer, submission) > -1:
                    ct += 1
            if ct < self.lowest_load[reviewer] or ct == 0:
                rev_to_ignore.append(reviewer)

        if rev_to_ignore:
            rev_to_ignore.sort(reverse=True)

            for reviewer in rev_to_ignore:
                self.similarity = np.delete(self.similarity, reviewer, 0)
                del self.aca_ind[reviewer]
                del self.location[reviewer]
                del self.seniority[reviewer]
                del self.ability[reviewer]
                del self.lowest_load[reviewer]

                self.ignored_reviewers.append(self.rev_id_to_key[reviewer])

                del self.rev_id_to_key[reviewer]

                # dependencies with PC
                new_dependant_revs = set()
                for pair in self.dependant_revs:
                    if pair[0] == reviewer or pair[1] == reviewer:
                        continue
                    elif pair[0] > reviewer > pair[1]:
                        new_dependant_revs.add((pair[0] - 1, pair[1]))
                    elif pair[1] > reviewer > pair[0]:
                        new_dependant_revs.add((pair[0], pair[1] - 1))
                    elif pair[0] > reviewer and pair[1] > reviewer:
                        new_dependant_revs.add((pair[0] - 1, pair[1] - 1))
                    else:
                        new_dependant_revs.add((pair[0], pair[1]))

                self.dependant_revs = new_dependant_revs

                # dependencies with ERC
                # modify dependencies of all other ERC reviewers if they contain the dropped reviewer
                for r in range(len(self.dependant_revs_erc)):
                    if r != reviewer:
                        new_r_d = []
                        for r_d in self.dependant_revs_erc[r]['pc']:
                            if r_d < reviewer:
                                new_r_d.append(r_d)
                            elif r_d > reviewer:
                                new_r_d.append(r_d - 1)
                            elif r_d == reviewer:
                                continue

                        self.dependant_revs_erc[r]['pc'] = new_r_d

                self.num_rev -= 1

    # main routine
    def assignment(self):
        if self.sim_threshold > 0:
            self.calculate_thresholded_similarities()

        # test if every reviewer has at least lower bound papers where similarity > similarity threshold
        # drop reviewers which have no possibility of being assigned to a submission
        self.test_reviewer_loads()

        # initially set diversity tracking variables
        bg_a = [0, 0, 0]
        bg_l = [0, 0, 0, 0, 0, 0, 0]
        bg_s = [0, 0, 0]

        for r in range(self.num_rev):
            bg_a[self.aca_ind[r]] += 1
            for c_l in self.location[r]:
                bg_l[self.location_ids[c_l]] += 1
            bg_s[self.seniority[r]] += 1

        bg_a[0] += bg_a[2]
        bg_a[1] += bg_a[2]

        # check if current reviewer pool is suitable in seniority perspective
        self.f.write('check seniority \n')
        suitable, true_abilities = self.check_seniority()

        if not suitable:
            # add new reviewers from missing backgrounds
            bg_a, bg_l, bg_s = self.add_new_reviewers_based_on_seniority(true_abilities, bg_a, bg_l, bg_s)

        self.f.write('seniority done \n')
        self.f.flush()

        # check if current reviewer pool is suitable in professional background perspective
        self.f.write('check professional background \n')

        suitable, true_abilities = self.check_professional_background()

        if not suitable:
            # add new reviewers from missing backgrounds
            bg_a, bg_l, bg_s = self.add_new_reviewers_based_on_professional_background(true_abilities, bg_a, bg_l, bg_s)

        self.f.write('professional background done \n')
        self.f.flush()

        # first run of algorithm with full PC and papers to check if dropping of edges will be the problem or the
        # current reviewers themselves
        suitable = self._assignment(1, drop_edges=False)

        self.f.write('first try did find a suitable assignment? ' + str(suitable) + '\n')
        self.f.flush()

        changes = True
        drop_new_papers = True

        while not suitable and len(self.similarity_erc) > 0 and changes:
            changes = False
            if drop_new_papers:
                print('new run where new rev are computed. ' + str(len(self.similarity_erc)) + ' rev left to include')
                self.f.write('new run where new rev are computed. ' + str(len(self.similarity_erc)) +
                             ' rev left to include\n')
                self.f.flush()
            else:
                print('no assignment was found, try again to include reviewers independent of problematic papers. ' +
                      str(len(self.similarity_erc)) + ' rev left to include')
                self.f.write('no assignment was found,  try again to include reviewers independent of problematic '
                             'papers. ' + str(len(self.similarity_erc)) + ' rev left to include\n')
                self.f.flush()

            # drop percentage of papers to check if they are the problem
            dp_assignments = []
            dp_papers = []

            # find possibly problematic papers to drop
            needed_rev_num_per_prob_sub = {}
            def_problematic_papers = []
            prob_problematic_papers = {}

            new_rev_candidates = {}

            if drop_new_papers:

                for submission in range(self.num_papers):
                    curr_slice = self.similarity[:, submission]
                    # count how many reviewers do not have COIs, if not enough reviewers remain, add paper to definitely
                    # problematic ones
                    ns = np.sum(np.array(curr_slice) != -1, axis=0)
                    if ns < self.demand:
                        def_problematic_papers.append(submission)
                        needed_rev_num_per_prob_sub[submission] = self.demand - ns
                    else:
                        prob_problematic_papers[submission] = float(sum(x for x in curr_slice if x != -1) / ns)

                    # handle definitely problematic papers
                bg_a, bg_l, bg_s, prob_problematic_papers = self.insert_for_def_problematic_papers(
                    def_problematic_papers, prob_problematic_papers, needed_rev_num_per_prob_sub, bg_a, bg_l, bg_s)

                # check if new reviewers have solved all issues, stop inclusion of new reviewers
                suitable = self._assignment(1, drop_edges=False)

                if suitable:
                    break

                # handle probably problematic papers
                prob_problematic_papers = dict(sorted(prob_problematic_papers.items(), key=lambda item: item[1]))
                observed_papers = []

                for sub in prob_problematic_papers:
                    observed_papers.append(sub)

                # find possibly problematic papers by doing several assignments where part of papers is dropped
                for i in range(0, 25):
                    dropped_papers = find_papers_to_drop(i, observed_papers)
                    dp_papers.append(dropped_papers)

                    subroutine_assignment = DiveRS_sub.DiveRSSub(self.similarity, self.dependant_revs, self.aca_ind,
                                                                 self.location, self.seniority, dropped_papers,
                                                                 self.demand, self.ability, 0, self.function)
                    subroutine_assignment.initialize_model()
                    dp_assignments.append(subroutine_assignment.assignment_step_subroutine())

                paper_list = list(range(self.num_papers))

                ct_of_papers = {}
                problematic_papers = {}
                for p in range(self.num_papers):
                    problematic_papers[p] = 0
                    ct_of_papers[p] = 0

                for run in range(len(dp_assignments)):
                    for p in paper_list:
                        if p not in dp_papers[run]:
                            ct_of_papers[p] += 1
                            if len(dp_assignments[run][0]) == 0:
                                problematic_papers[p] += 1

                for p in range(self.num_papers):
                    if ct_of_papers[p] > 0:
                        problematic_papers[p] = problematic_papers[p] / ct_of_papers[p]
                    else:
                        problematic_papers[p] = 0

                # iterate over problematic papers to find those which require new reviewers
                problematic_papers = sorted(problematic_papers.items(), key=lambda x: x[1], reverse=True)

                ten_most_prob_papers = []
                ct = 0

                curr = []
                pp: Tuple[int, float]
                for pp in problematic_papers:
                    if pp not in self.ignored_prob_papers:
                        if ct < 10:
                            ct += 1
                            if pp[1] > 0.5:
                                ten_most_prob_papers.append(pp[0])
                                curr.append(pp)

                # iterate over problematic papers from last try to identify ones for which no reviewers could be found,
                # try to assign reviewers only to those single papers, if this does not work, new reviewers will be
                # assigned to the PC for it the paper will be considered as out of scope
                last_pp: Tuple[int, float]
                for last_pp in self.problematic_papers_from_last_try:
                    if last_pp in curr and last_pp[0] not in self.ignored_prob_papers and last_pp[1] == 1.0:
                        if last_pp[1] == 1.0:
                            dropped_papers = list(range(self.num_papers))
                            dropped_papers.remove(last_pp[0])

                            subroutine_assignment = DiveRS_sub.DiveRSSub(self.similarity, self.dependant_revs,
                                                                         self.aca_ind, self.location, self.seniority,
                                                                         dropped_papers, self.demand, self.ability, 0,
                                                                         self.function)
                            subroutine_assignment.initialize_model()
                            assignment, found = subroutine_assignment.assignment_step_subroutine()

                            # if assignment can be found for the single paper, ignore it in allocation of new reviewers
                            if found:
                                self.ignored_prob_papers.append(last_pp[0])
                            else:
                                # include new rev for submission
                                ten_most_prob_papers.remove(last_pp[0])
                                candidate_count = 0
                                cand_lst: List[Tuple[int, float]] = []
                                for new_rev in list(range(len(self.similarity_erc))):
                                    if self.similarity_erc.item(new_rev, last_pp) != -1:
                                        candidate_count += 1
                                        cand_lst.append((new_rev, self.similarity_erc.item(new_rev, last_pp)))

                                if candidate_count > 0:
                                    new_rev_candidates_curr = dict(
                                        sorted(cand_lst.items(), key=lambda item: item[1], reverse=True))

                                    inserted_candidates_curr = []

                                    for candidate_id in new_rev_candidates_curr:
                                        if len(inserted_candidates_curr) < 3:
                                            changes = True
                                            # insert new candidate
                                            inserted_candidates_curr[candidate_id] = len(self.aca_ind)

                                            self.include_new_reviewer(candidate_id)

                                            # update background variables
                                            bg_a, bg_l, bg_s = self.update_bg_variables(candidate_id, bg_a, bg_l, bg_s)
                                        elif len(inserted_candidates_curr) == 3:
                                            break
                                else:
                                    self.check_if_paper_should_be_ignored(last_pp)
                self.problematic_papers_from_last_try = curr
            else:
                # run where everything should be good (no specifically problematic manuscripts were found) but isn't ->
                # include new reviewers fitting manuscripts the most and recheck
                ten_most_prob_papers = list(range(self.num_papers))

            for pp in ten_most_prob_papers:
                # find new reviewers for this specific problematic paper
                # update list of fitting reviewers for all papers
                candidate_count = 0
                for new_rev in list(range(len(self.similarity_erc))):
                    if self.similarity_erc.item(new_rev, pp) != -1:
                        candidate_count += 1

                if candidate_count > 0:
                    for new_rev in range(len(self.similarity_erc)):
                        # if new candidate is not already getting included and has no conflict of interest, they can be
                        # looked at
                        if self.similarity_erc.item(new_rev, pp) != -1:
                            cond_1 = self.sim_threshold > 0
                            cond_2 = cond_1 and self.similarity_erc.item(new_rev, pp) >= self.sim_threshold
                            if not cond_1 or cond_2:
                                if new_rev in new_rev_candidates:
                                    new_rev_candidates[new_rev].append(self.similarity_erc.item(new_rev, pp))
                                else:
                                    new_rev_candidates[new_rev] = [self.similarity_erc.item(new_rev, pp)]
                else:
                    self.check_if_paper_should_be_ignored(pp)

            # bias list of fitting reviewers such that underrepresented diversity aspects are valued more
            bias_l = [0, 0, 0, 0, 0, 0, 0]
            bias_s = [0, 0, 0]

            loc_dic = {}
            sen_dic = {}
            for c_l in range(7):
                loc_dic[c_l] = bg_l[c_l]
            for c_s in range(3):
                sen_dic[c_s] = bg_s[c_s]

            loc_dic: Dict[Any, Any] = dict(sorted(loc_dic.items(), key=lambda item: item[1], reverse=True))
            sen_dic: Dict[Any, Any] = dict(sorted(sen_dic.items(), key=lambda item: item[1], reverse=True))

            factor = 3
            for c_l in loc_dic:
                bias_l[c_l] = 0.25 / 10.5 * factor
                factor -= 0.5

            factor = 1
            for c_s in sen_dic:
                bias_s[c_s] = 0.25 / 1.5 * factor
                factor -= 0.5

            for rc in new_rev_candidates:
                bias = 1

                # professional background
                if bg_a[0] < bg_a[1] and self.aca_ind_erc[rc] == 0:
                    bias += 0.25
                if bg_a[0] > bg_a[1] and self.aca_ind_erc[rc] == 1:
                    bias += 0.25
                if self.aca_ind_erc[rc] == 2:
                    bias += 0.25

                # location
                for l_c in self.location_erc[rc]:
                    bias += bias_l[self.location_ids[l_c]]

                # seniority
                bias += bias_s[self.seniority_erc[rc]]

                new_rev_candidates[rc] = bias * sum(new_rev_candidates[rc])

            new_rev_candidates = dict(sorted(new_rev_candidates.items(), key=lambda item: item[1], reverse=True))

            # add new reviewers to PC
            inserted_candidates = {}

            for candidate_id in new_rev_candidates:
                if len(inserted_candidates) < self.kappa:
                    if self.sim_threshold > 0:
                        ct = 0
                        for submission in range(self.num_papers):
                            if self.similarity_erc.item(candidate_id, submission) >= self.sim_threshold:
                                ct += 1

                        if ct > 0:
                            changes = True
                            # insert new candidate
                            inserted_candidates[candidate_id] = len(self.aca_ind)

                            self.include_new_reviewer(candidate_id)

                            # update background variables
                            bg_a, bg_l, bg_s = self.update_bg_variables(candidate_id, bg_a, bg_l, bg_s)
                    else:
                        changes = True
                        # insert new candidate
                        inserted_candidates[candidate_id] = len(self.aca_ind)

                        self.include_new_reviewer(candidate_id)

                        # update background variables
                        bg_a, bg_l, bg_s = self.update_bg_variables(candidate_id, bg_a, bg_l, bg_s)
                elif len(inserted_candidates) == self.kappa:
                    break

            # delete new reviewers from ERC
            inserted_candidates = collections.OrderedDict(sorted(inserted_candidates.items(), reverse=True))

            for i_c in inserted_candidates:
                self.drop_reviewer_from_erc(i_c, inserted_candidates[i_c])

            self.num_rev = self.num_rev + len(inserted_candidates)

            if self.sim_threshold > 0:
                self.calculate_thresholded_similarities()

            suitable = self._assignment(1, drop_edges=False)

            if len(ten_most_prob_papers) == 0 and not suitable:
                drop_new_papers = False
                changes = True
        if not suitable:
            raise ValueError('No reviewer set could be found such that an assignment could be computed for the current '
                             'set of submissions.')

        self.f.write('insertion of new reviewers completed \n')
        self.f.flush()
        # actual run which will deliver the best possible assignment out of all tries which are conducted
        self._assignment(self.tries)

    # pp - id of currently problematic paper
    def check_if_paper_should_be_ignored(self, pp):
        # there are no more reviewers to include for this submission

        # test if any of the existing reviewers are even able to review submission, disregard all other manuscripts for
        # a moment, run assignment only with currently problematic paper
        dropped_papers = list(range(self.num_papers))
        dropped_papers.remove(pp)

        subroutine_assignment = DiveRS_sub.DiveRSSub(self.similarity, self.dependant_revs, self.aca_ind,
                                                     self.location, self.seniority, dropped_papers,
                                                     self.demand, self.ability, 0, self.function)

        subroutine_assignment.initialize_model()
        assignment, found = subroutine_assignment.assignment_step_subroutine()

        # if no assignment can be found for the single paper, mark it as out of scope, ignore it;
        # if assignment can be found no new reviewers it is marked as a problematic paper to ignore
        if found:
            self.ignored_prob_papers.append(pp)
        else:
            self.delete_oos_papers([pp], [])

    def calculate_diversity_score(self, assignment):
        diversities = []

        for p in assignment:
            diversity = 0
            # professional background
            # = 0: industry, = 1: academia, = 2: both
            prof_bg = 0
            for r in assignment[p]:
                if self.aca_ind[r] == 0:
                    prof_bg += -1
                if self.aca_ind[r] == 1:
                    prof_bg += 1

            diversity += 1 - (abs(prof_bg) / self.demand)

            # location
            loc = 0
            reviewer_pairs = list(itertools.combinations(assignment[p], 2))
            for pair in reviewer_pairs:
                intersection = [val for val in self.location[pair[0]] if val in self.location[pair[1]]]
                union = set(self.location[pair[0]] + self.location[pair[1]])

                loc += len(intersection) / len(union)

            diversity += 1 - (1 / len(reviewer_pairs)) * loc

            # seniority
            check = set()
            for r in assignment[p]:
                check.add(self.seniority[r])

            diversity += 1 / 3 * len(check)
            diversities.append(diversity)

        return sum(diversities) / len(diversities)


# assignment - reviewer-manuscript assignment which should be evaluated
# aca_ind - dict of reviewer to information whether this person works in industry or academia or both
# location - dict of reviewer to list of continents, in which they work
# seniority - dict of reviewer to their seniority
# demand - number of reviewers assigned to each manuscript
def calculate_diversity_score(assignment, aca_ind, location, seniority, demand):
    diversities = []

    for p in assignment:
        diversity = 0
        # professional background
        # = 0: industry, = 1: academia, = 2: both
        prof_bg = 0
        for r in assignment[p]:
            if aca_ind[r] == 0:
                prof_bg += -1
            if aca_ind[r] == 1:
                prof_bg += 1

        diversity += 1 - (abs(prof_bg) / demand)

        # location
        loc = 0
        reviewer_pairs = list(itertools.combinations(assignment[p], 2))
        for pair in reviewer_pairs:
            intersection = [val for val in location[pair[0]] if val in location[pair[1]]]
            union = set(location[pair[0]] + location[pair[1]])

            loc += len(intersection) / len(union)

        diversity += 1 - (1 / len(reviewer_pairs)) * loc

        # seniority
        check = set()
        for r in assignment[p]:
            check.add(seniority[r])

        diversity += 1 / 3 * len(check)
        diversities.append(diversity)

    return sum(diversities) / len(diversities)
