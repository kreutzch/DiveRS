from itertools import product

from gurobipy import Model, GRB
import numpy as np


# thanks to Stelmakh et al. for providing the basic structure of the code: https://www.cs.cmu.edu/~istelmak/


# used for the restricted network when papers are dropped from the original network
class DiveRSSub:
    # tolerance for integrality check
    _EPS = 1e-3

    # initialize the parameters
    # similarity - similarity matrix between reviewers and papers with COIs coded into as -1
    # dependant_revs - pairs of reviewers which are not independent from each other
    # aca_ind - dict of reviewer to information whether this person works in industry or academia or both
    #           aca_ind[reviewer] : 2 = works in both, 1 = works in aca, 0 = works in ind
    # location - dict of reviewer to list of continents, in which they work
    #            location[reviewer] contains 0 = 'SA', 1 = 'AF', 2 = 'AN', 3 = 'AS', 4 = 'OC', 5 = 'NA', 6 = 'EU'
    # seniority - dict of reviewer to their seniority
    #            seniority[reviewer] : 0 = senior, 1 = advanced, 2 = junior
    # deleted_papers - papers for which not enough reviewers could be found in the original PC or ERC, our of scope
    #                  papers
    # demand - lambda, requested number of reviewers per paper
    # ability - the maximum number of papers a reviewer can review
    # lowest_load - the minimum number of papers a reviewer needs to review
    # function - transformation function for similarities
    # delta - accepted margin of error for max-flow calculation due to floating point addition
    def __init__(self, similarity, dependant_revs, aca_ind, location, seniority, deleted_papers, demand, ability,
                 lowest_load, function=lambda x: x, delta=1e-5):
        self._problem_subroutine = None
        self._balance_dependencies = None
        self._balance_decisions_sen = None
        self._balance_decisions_loc = None
        self._balance_decisions_aca = None
        self._balance_papers_sen = None
        self._balance_papers_loc = None
        self._balance_papers_aca = None
        self._cond_decision = None
        self._balance_papers_a2 = None
        self._balance_papers = None
        self._balance_diversity = None
        self._balance_reviewers = None
        self._diversity_vars = None
        self._hlp_vars = None
        self._decision_vars = None
        self.source_vars = None
        self._sink_vars = None

        for del_p in sorted(deleted_papers, reverse=True):
            similarity = np.delete(similarity, del_p, 1)

        self.similarity = similarity
        self.num_rev = int(similarity.shape[0])
        self.num_papers = int(similarity.shape[1])

        self.ability = ability
        try:
            ll = int(lowest_load)
            self.lowest_load = []
            for p in range(self.num_rev):
                self.lowest_load.append(ll)
        except TypeError:
            self.lowest_load = lowest_load
        self.demand = demand

        self.function = function
        self.dependant_revs = dependant_revs

        self.aca_ind = aca_ind
        self.location_ids = {'SA': 0, 'AF': 1, 'AN': 2, 'AS': 3, 'OC': 4, 'NA': 5, 'EU': 6}
        self.location = location
        self.seniority = seniority

        self.delta = delta

    # initializes the flow network
    def initialize_model(self):
        problem = Model()

        # verbose yes or no
        problem.setParam('OutputFlag', False)

        problem.params.NonConvex = 2

        # layers:
        # L1: source
        # L2: reviewers
        # L3: decision vertex: which reviewer with which paper
        # L4: diversity: professional background, location, seniority
        # L5: papers
        # L6: sink

        # edges from source to reviewers, capacity controls maximum reviewer load
        # edges between layer L1 and layer L2
        self.source_vars = problem.addVars(self.num_rev, vtype=GRB.INTEGER, lb=0, ub=0, name='reviewers')

        for r in range(self.num_rev):
            self.source_vars[r].ub = self.ability[r] * 3
            self.source_vars[r].lb = self.lowest_load[r] * 3

        # edges from papers to sink, capacity controls a number of reviewers per paper
        # edges between layer L5 and layer L6
        self._sink_vars = problem.addVars(self.num_papers, vtype=GRB.INTEGER, lb=0, ub=0, name='papers')

        # edges between reviewers and decision
        # edges between layer L2 and layer L3
        self._decision_vars = problem.addVars(self.num_rev, self.num_rev * self.num_papers, vtype=GRB.INTEGER, lb=0,
                                              ub=0, name='decision')

        # edges between decision and diversity. Initially capacities are set to 0 (no edge is added in the network)
        # edges between layer L3 and layer L4
        # industry/academia: 3 vertices per paper, location: 14 vertices per paper, seniority: 3 vertices
        self._hlp_vars = problem.addVars(self.num_rev * self.num_papers, self.num_papers * 20, vtype=GRB.CONTINUOUS,
                                         lb=0.0, ub=0.0, name='hlp')

        # edges between diversity and papers
        # edges between layer L4 and layer L5
        # industry/academia: 3 vertices per paper, location: 14 vertices per paper, seniority: 3 vertices
        self._diversity_vars = problem.addVars(self.num_papers * 20, self.num_papers, vtype=GRB.CONTINUOUS, lb=0.0,
                                               ub=0.0, name='diversity')

        # flow balance equations for reviewer (source) - decision
        self._balance_reviewers = problem.addConstrs((self.source_vars[i] == self._decision_vars.sum(i, '*')
                                                      for i in range(self.num_rev)))

        # flow balance equations for mix - diversity
        self._balance_diversity = problem.addConstrs((self._hlp_vars.sum('*', i) == (self._diversity_vars.sum(i, '*'))
                                                      for i in range(self.num_papers * 20)))

        # flow balance equations for diversity - papers (sink)
        self._balance_papers = problem.addConstrs((self._sink_vars[i] == (self._diversity_vars.sum('*', i))
                                                   for i in range(self.num_papers)))

        self._balance_papers_a2 = problem.addConstrs((self._sink_vars[i] == self.demand * 3) for i in
                                                     range(self.num_papers))

        # flow balance equation for decision variables: variable needs to be either 1 or 0
        self._cond_decision = problem.addConstrs((0 == (self._decision_vars[i, j] - 3) * self._decision_vars[i, j]
                                                  for i in range(self.num_rev) for j in
                                                  range(self.num_rev * self.num_papers)))

        self._balance_papers_aca = problem.addConstrs((self.demand == self._diversity_vars[i * 20, i] +
                                                       self._diversity_vars[i * 20 + 1, i] +
                                                       self._diversity_vars[i * 20 + 2, i]
                                                       for i in range(self.num_papers)))

        self._balance_papers_loc = problem.addConstrs((self.demand == self._diversity_vars[i * 20 + 3, i] +
                                                       self._diversity_vars[i * 20 + 4, i] +
                                                       self._diversity_vars[i * 20 + 5, i] +
                                                       self._diversity_vars[i * 20 + 6, i] +
                                                       self._diversity_vars[i * 20 + 7, i] +
                                                       self._diversity_vars[i * 20 + 8, i] +
                                                       self._diversity_vars[i * 20 + 9, i] +
                                                       self._diversity_vars[i * 20 + 10, i] +
                                                       self._diversity_vars[i * 20 + 11, i] +
                                                       self._diversity_vars[i * 20 + 12, i] +
                                                       self._diversity_vars[i * 20 + 13, i] +
                                                       self._diversity_vars[i * 20 + 14, i] +
                                                       self._diversity_vars[i * 20 + 15, i] +
                                                       self._diversity_vars[i * 20 + 16, i]
                                                       for i in range(self.num_papers)))

        self._balance_papers_sen = problem.addConstrs((self.demand == self._diversity_vars[i * 20 + 17, i] +
                                                       self._diversity_vars[i * 20 + 18, i] +
                                                       self._diversity_vars[i * 20 + 19, i]
                                                       for i in range(self.num_papers)))

        # flow balance equations for decision - hlp
        self._balance_decisions_aca = problem.addConstrs(self._decision_vars[r, r * self.num_papers + p] / 3 ==
                                                         self._hlp_vars[r * self.num_papers + p, p * 20] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 1] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 2]
                                                         for r in range(self.num_rev) for p in range(self.num_papers))

        self._balance_decisions_loc = problem.addConstrs(self._decision_vars[r, r * self.num_papers + p] / 3 ==
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 3] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 4] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 5] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 6] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 7] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 8] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 9] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 10] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 11] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 12] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 13] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 14] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 15] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 16]
                                                         for r in range(self.num_rev) for p in range(self.num_papers))

        self._balance_decisions_sen = problem.addConstrs(self._decision_vars[r, r * self.num_papers + p] / 3 ==
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 17] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 18] +
                                                         self._hlp_vars[r * self.num_papers + p, p * 20 + 19]
                                                         for r in range(self.num_rev) for p in range(self.num_papers))

        # flow balance constraints to prevent dependent reviewers in a set: simplified XOR
        # idea: a + b == a + b - ab
        self._balance_dependencies = []
        for pn in range(0, self.num_papers):
            for pair in self.dependant_revs:
                # these normal constraints are cast to qconstraints
                self._balance_dependencies.append(
                    problem.addConstr((self._decision_vars[pair[0], pair[0] * self.num_papers + pn] +
                                       self._decision_vars[pair[1], pair[1] * self.num_papers + pn]) ==
                                      (self._decision_vars[pair[0], pair[0] * self.num_papers + pn] +
                                       self._decision_vars[pair[1], pair[1] * self.num_papers + pn]) -
                                      (self._decision_vars[pair[0], pair[0] * self.num_papers + pn] *
                                       self._decision_vars[pair[1], pair[1] * self.num_papers + pn]), 'balance_const'))

        self._problem_subroutine = problem

    # delete edges with conflicts of interests from network
    # pairs - all pairs of reviewers and papers as edges
    # dropped_edges - list of reviewer-paper edges to exclude in the network
    def _cleanse_pairs_tiny(self, pairs, dropped_edges):
        pairs_wo_dropped_edges = []

        for curr_pair in pairs:
            if curr_pair not in dropped_edges and self.similarity[curr_pair[0], curr_pair[1]] > -1:
                pairs_wo_dropped_edges.append(curr_pair)

        return pairs_wo_dropped_edges

    # actual reviewer assignment step for all papers except those in dropped_papers
    # pairs - all combinations of reviewer-paper edges
    # dropped_edges - edges which need to be dropped
    def assignment_step_subroutine(self, dropped_edges=None):

        if dropped_edges is None:
            dropped_edges = []
        pairs = [[reviewer, paper] for (reviewer, paper) in product(range(self.num_rev), range(self.num_papers))]
        # set up the max flow objective
        self._problem_subroutine.reset()
        self._problem_subroutine.setObjective(sum([self.source_vars[i] for i in range(self.num_rev)]), GRB.MAXIMIZE)

        # reset all variables in the flow network
        self._reset_variables()

        ################################################################################################################
        # adjust all variables to current values for background
        ################################################################################################################

        # adjust background vertices' load
        for paper in range(self.num_papers):
            for i in range(0, 20):
                # i = 0, 1, 2 : aspect industry/academia, 2 = both
                # i = 3, 4, 5, 6, 7, 8, 9 : aspect location ("yes" option for continents)
                # i = 10, 11, 12, 13, 14, 15, 16 : aspect location ("no" option for continents)
                # i = 17, 18, 19 : aspect seniority, 17 (0) = senior

                # case i = 0: industry, i = 1: academia
                if i < 2:
                    self._diversity_vars[paper * 20 + i, paper].ub = (self.demand - 1)
                # case i = 2: both
                elif i == 2:
                    self._diversity_vars[paper * 20 + i, paper].ub = self.demand
                    self._diversity_vars[paper * 20 + i, paper].lb = 0
                # case i = 3 to 9: continents "yes"-vertex option
                elif i < 10:
                    self._diversity_vars[paper * 20 + i, paper].ub = (self.demand - 1) / 7
                # case i = 10 to 16: continents "no"-vertex option
                elif i < 17:
                    self._diversity_vars[paper * 20 + i, paper].ub = self.demand / 7
                # case i = 17: senior
                elif i == 17:
                    self._diversity_vars[paper * 20 + i, paper].lb = 1
                    self._diversity_vars[paper * 20 + i, paper].ub = self.demand
                # case i > 17: advanced and junior
                else:
                    self._diversity_vars[paper * 20 + i, paper].ub = self.demand - 1

        # delete edges with COIs and those we want to drop from all pairs
        cleansed_pairs = self._cleanse_pairs_tiny(pairs, dropped_edges)

        for curr_pair in cleansed_pairs:
            self._decision_vars[curr_pair[0], curr_pair[0] * self.num_papers + curr_pair[1]].ub = 3
            # set possible load for professional background
            self._hlp_vars[curr_pair[0] * self.num_papers + curr_pair[1], curr_pair[1] * 20 + self.aca_ind[
                curr_pair[0]]].ub = 1

            # set possible loads for current location ("yes"-vertex options), also set load for continents which are
            # not part of current location ("no"-vertex options)
            curr_locations = []
            for c_l in self.location[curr_pair[0]]:
                curr_locations.append(self.location_ids[c_l])

            for i in range(0, 7):
                if i in curr_locations:
                    self._hlp_vars[
                        curr_pair[0] * self.num_papers + curr_pair[1], curr_pair[1] * 20 + 3 + i].ub = 1 / 7
                else:
                    self._hlp_vars[
                        curr_pair[0] * self.num_papers + curr_pair[1], curr_pair[1] * 20 + 10 + i].ub = 1 / 7

            # set possible load for seniority
            self._hlp_vars[curr_pair[0] * self.num_papers + curr_pair[1], curr_pair[1] * 20 + 17 + self.seniority[
                curr_pair[0]]].ub = 1

        # adjust reviewer vertices' loads in the network
        for reviewer in range(self.num_rev):
            self.source_vars[reviewer].ub = self.ability[reviewer] * 3
            self.source_vars[reviewer].lb = self.lowest_load[reviewer] * 3

        ################################################################################################################
        # calculate and return assignment which results in max flow
        ################################################################################################################

        # max cost max flow objective: sum similarity of reviewer set assigned to paper needs maximal in general
        self._problem_subroutine.setObjective(
            sum([sum([self.similarity[reviewer, paper] * self._decision_vars[reviewer, reviewer *
                                                                             self.num_papers +
                                                                             paper] / 3
                      for paper in range(self.num_papers)]) for reviewer in
                 range(self.num_rev)]), GRB.MAXIMIZE)

        self._problem_subroutine.update()
        self._problem_subroutine.optimize()

        # compute actual assignment, this could throw an AttributeError exception if no max flow can be found
        try:
            maxflow = self._problem_subroutine.objVal

            if maxflow:
                # return assignment
                assignment = {}

                # initialise assignment of reviewers as empty for all papers
                for paper in range(self.num_papers):
                    assignment[paper] = []

                # if decision variable indicates that reviewer is assigned to a paper, assign reviewer to paper
                for reviewer in range(self.num_rev):
                    for paper in range(self.num_papers):
                        if self._decision_vars[reviewer, reviewer * self.num_papers + paper].X / 3 + self.delta >= 1:
                            assignment[paper] += [reviewer]

                        if np.abs(self._decision_vars[reviewer, reviewer * self.num_papers + paper].X - int(
                                self._decision_vars[reviewer, reviewer * self.num_papers + paper].X + self.delta)) > \
                                self._EPS:
                            raise ValueError('Error with rounding -- please check that demand and ability are integer.')

                return assignment, True
        except AttributeError:
            return [], False

    # reset network flow variables
    def _reset_variables(self):
        # reset assignment: decision - background weights
        for reviewer in range(self.num_rev):
            for decision in range(self.num_rev * self.num_papers):
                self._decision_vars[reviewer, decision].lb = 0
                self._decision_vars[reviewer, decision].ub = 0

        # reset hlp: decision - background weights
        for r in range(self.num_rev):
            for paper in range(self.num_papers):
                for p in range(self.num_papers):
                    for i in range(0, 20):
                        self._hlp_vars[r * self.num_papers + p, paper * 20 + i].lb = 0
                        self._hlp_vars[r * self.num_papers + p, paper * 20 + i].ub = 0

        # reset sinks: assign paper with exactly kappa reviewers
        for paper in range(self.num_papers):
            # if paper is dropped from the network (in the runs where the papers are identifies which need other/more
            # reviewers) it cannot have flow

            self._sink_vars[paper].ub = self.demand * 3
            self._sink_vars[paper].lb = 0
