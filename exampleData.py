import numpy as np
from sklearn import metrics


# this method will fill the required data structures with example data
def fetch_example_data():
    # data structures for original PC members
    # similarity - similarity matrix between reviewers and submissions, S
    # dependant_revs - pairs of ids of reviewers which are not independent of each other
    # aca_ind - dict of [reviewer, paper] to information whether this person works in industry or academia or both
    # location - list of continents, in which reviewers work
    # seniority - list indicating seniority of reviewer with id = position
    # ability - max number of papers reviewer can review, can be different for everyone, my^u
    # lowest_load - min number of papers reviewer has to review, can be different for everyone, my^l
    # rev_id_to_key - ids of PC members

    # data structures for ERC members
    # similarity_erc
    # dependant_revs_erc
    # aca_ind_erc
    # location_erc
    # seniority_erc
    # ability_erc
    # lowest_load_erc
    # rev_id_to_key_erc
    ########################################################################

    # vector representations of manuscripts as well as reviewers, vector gives probability distribution for 5 dimensions
    manuscripts = [[0.5, 0.1, 0.2, 0.1, 0.1],
                   [0, 0, 0, 0.5, 0.5],
                   [0.3, 0.2, 0.4, 0.1, 0]]
    reviewers = [[0.3, 0.2, 0.1, 0.2, 0.2],
                 [0, 0, 0, 1, 0],
                 [0.1, 0.3, 0.2, 0.3, 0.1],
                 [0.3, 0.2, 0.1, 0, 0.4],
                 [0.9, 0.05, 0.025, 0.05, 0.025],
                 [0.1, 0.1, 0.4, 0, 0.4]]
    reviewers_erc = [[0, 0.4, 0.1, 0.4, 0.1],
                     [0.25, 0.25, 0.25, 0.15, 0.1],
                     [0.2, 0.3, 0.1, 0.3, 0.1],
                     [0.2, 0.2, 0.2, 0.2, 0.2]]

    # for each reviewer it needs to be stored with which manuscript they have COIs
    cois = [[1], [], [0, 1, 2], [], [1, 2], []]
    cois_erc = [[], [], [], [1]]

    # calculation of similarity between all manuscripts and reviewers with consideration of COIs
    # dimensions: n (number of reviewers) * m (number of submissions), with COIs between reviewers and authors of
    # manuscripts coded into matrix as -1

    similarity = []
    for i in range(len(reviewers)):
        sim_for_rev = []
        for j in range(len(manuscripts)):
            if j not in cois[i]:
                sim_for_rev.append(metrics.pairwise.cosine_similarity([reviewers[i]], [manuscripts[j]])[0][0])
            else:
                sim_for_rev.append(-1)
        similarity.append(sim_for_rev)
    similarity_erc = []
    for i in range(len(reviewers_erc)):
        sim_for_rev = []
        for j in range(len(manuscripts)):
            if j not in cois_erc[i]:
                sim_for_rev.append(metrics.pairwise.cosine_similarity([reviewers_erc[i]], [manuscripts[j]])[0][0])
            else:
                sim_for_rev.append(-1)
        similarity_erc.append(sim_for_rev)

    similarity = np.array(similarity)
    similarity_erc = np.array(similarity_erc)

    dependant_revs = [(0, 1), (1, 0), (1, 2), (2, 1), (3, 1), (1, 3)]
    dependant_revs_erc = [{'pc': [1, 4], 'ERC': [1]}, {'pc': [], 'ERC': [0]}, {'pc': [3], 'ERC': []},
                          {'pc': [], 'ERC': []}]

    # 0 = ind, 1 = aca, 2 = both
    aca_ind = [1, 0, 1, 0, 1, 2]
    aca_ind_erc = [1, 0, 2, 2]

    # {'SA': 0, 'AF': 1, 'AN': 2, 'AS': 3, 'OC': 4, 'NA': 5, 'EU': 6}
    location = [['SA'], ['AF', 'SA'], ['AF'], ['AF', 'EU'], ['AF'], ['SA']]
    location_erc = [['OC'], ['EU', 'SA'], ['EU'], ['AF']]

    # 0 = senior, 1 = advanced, 2 = junior
    seniority = [0, 1, 0, 0, 2, 1]
    seniority_erc = [0, 1, 2, 0]

    ability = [1, 1, 2, 3, 1, 1]
    ability_erc = [2, 2, 2, 2]

    lowest_load = [0, 0, 0, 0, 1, 0]
    lowest_load_erc = [0, 0, 0, 0]

    rev_id_to_key = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5']
    rev_id_to_key_erc = ['e0', 'e1', 'e2', 'e3']

    textual_representation_of_reviewers = {}
    for i in range(len(rev_id_to_key)):
        textual_representation_of_reviewers[rev_id_to_key[i]] = reviewers[i]
    for i in range(len(rev_id_to_key_erc)):
        textual_representation_of_reviewers[rev_id_to_key_erc[i]] = reviewers_erc[i]

    return [similarity, similarity_erc, dependant_revs_erc, dependant_revs, aca_ind, aca_ind_erc, location,
            location_erc, seniority, seniority_erc, ability, ability_erc, lowest_load, lowest_load_erc, rev_id_to_key,
            rev_id_to_key_erc, len(rev_id_to_key), textual_representation_of_reviewers]
