import numpy as np
import itertools
import sklearn


# computes fairness of an assignment -> see PR4All
# assignment - for all papers sets of their assigned reviewers
# function - function to transform similarity values
# similarity - similarity matrix between reviewers and papers
def fairness(assignment, function, similarity):
    curr_fairness = np.inf

    for paper in assignment:
        if curr_fairness > sum([function(similarity[reviewer, paper]) for reviewer in assignment[paper]]):
            curr_fairness = np.sum([function(similarity[reviewer, paper]) for reviewer in assignment[paper]])
    return curr_fairness


# prints fairness distribution for all assignments
# assignment - for all papers sets of their assigned reviewers
# function - function to transform similarity values
# similarity - similarity matrix between reviewers and papers
def fairness_distribution(assignment, function, similarity):
    for paper in assignment:
        print(sum([function(similarity[reviewer, paper]) for reviewer in assignment[paper]]))


# prints assignment to file
# assignment - for all papers sets of their assigned reviewers
# rev_id_to_key - reviewer ids to their unique keys which will be displayed
# file - target file
def print_assignment(assignment, rev_id_to_key, file):
    f = open(file, "w")

    for a in assignment:
        for rev in assignment[a]:
            f.write(rev_id_to_key[rev])
            f.write(',')
        f.write('\n')
        f.flush()

    f.close()


# calculates max, mean, mean (with original PC) number of assigned papers per reviewer, calculates number of unused ones
# diverse_assignment - for all papers sets of their assigned reviewers
# rev_id_to_key - reviewer ids to their unique keys which will be displayed
# ignored_reviewers - reviewers from original PC which are not assigned to any manuscripts
def calculate_max_mean_assignments(diverse_assignment, rev_id_to_key, ignored_reviewers):
    # find current PC
    pc_set = set()
    pc_ct = {}
    for manuscript in diverse_assignment:
        for rev in diverse_assignment[manuscript]:
            pc_set.add(rev_id_to_key[rev])
            if rev in pc_ct:
                pc_ct[rev] = pc_ct[rev] + 1
            else:
                pc_ct[rev] = 1

    pc_ct_l = list(pc_ct.values())

    return [max(pc_ct_l), sum(pc_ct_l) / len(pc_ct_l), sum(pc_ct_l) / (len(pc_set) + len(ignored_reviewers)),
            (len(pc_set) + len(ignored_reviewers)) - len(pc_ct_l)]


# computes dependencies between reviewers in assigned sets
# diverse_assignment - for all papers sets of their assigned reviewers
# dependant_revs - dependencies between reviewers
def comp_violated_dependencies(diverse_assignment, dependant_revs):
    # test how many dependencies are violated
    conflict = 0
    for assignment in diverse_assignment:
        found_conf = False
        for subset in itertools.combinations(diverse_assignment[assignment], 2):
            if (subset[0], subset[1]) in dependant_revs:
                found_conf = True
                break
        if found_conf:
            conflict += 1

    return conflict / len(diverse_assignment)


# calculates KL diversity of assignment
# diverse_assignment - for all papers sets of their assigned reviewers
# rev_id_to_key - reviewer ids to their unique keys which will be displayed
# textual_representation_of_reviewers - vector representation of reviewers, vectors need to be probability distributions
def calc_kl_diversity(diverse_assignment, rev_id_to_key, textual_representation_of_reviewers):
    all_kl_divs = []
    # iterate over all submissions
    for assignment in diverse_assignment:
        # iterate over pairs of assigned reviewers
        assignments_kl_divs = []
        for subset in itertools.combinations(diverse_assignment[assignment], 2):
            rev_a = textual_representation_of_reviewers[rev_id_to_key[subset[0]]]
            rev_b = textual_representation_of_reviewers[rev_id_to_key[subset[1]]]

            assignments_kl_divs.append(sklearn.metrics.mutual_info_score(rev_a, rev_b))
        all_kl_divs.append(sum(assignments_kl_divs) / len(assignments_kl_divs))

    return sum(all_kl_divs) / len(all_kl_divs)
