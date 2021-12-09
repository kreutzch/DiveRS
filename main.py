from typing import Any, Callable

import evaluate
import exampleData, DiveRS_main

if __name__ == '__main__':
    # general settings
    demand = 3  # lambda, number of required reviewers per submission
    theta = 0.3  # similarity threshold under which reviewers are not considered to be assigned for a submission
    tries = 25  # number of tries to find most diverse assignment in main routine
    kappa = 2  # max number of reviewers being included in one iteration of the main routine

    ################################################################################################################
    # fill data structures with example data
    [similarity, similarity_ERC, dependant_revs_ERC, dependant_revs, aca_ind, aca_ind_ERC, location, location_ERC,
     seniority, seniority_ERC, ability, ability_ERC, lowest_load, lowest_load_ERC, rev_id_to_key, rev_id_to_key_erc,
     initial_pc_size, textual_representation_of_reviewers] = exampleData.fetch_example_data()

    # transformation functions for similarities
    avg_est: Callable[[Any], Any] = lambda x: x
    mle_est: Callable[[Any], Any] = lambda x: 1. / (1 - x) if x < 1 else 1e6

    ################################################################################################################
    # calculate assignment
    print('Algorithm started')

    a = DiveRS_main.DiveRSMain(demand, similarity, ability, lowest_load, dependant_revs, aca_ind, location, seniority,
                               tries, similarity_ERC, ability_ERC, lowest_load_ERC, dependant_revs_ERC, aca_ind_ERC,
                               location_ERC, seniority_ERC, rev_id_to_key, rev_id_to_key_erc, theta, kappa,
                               function=mle_est, percentage=0.1)
    a.assignment()

    ################################################################################################################
    print('Evaluation started')
    # print assignment to file
    evaluate.print_assignment(a.diverse_assignment, a.rev_id_to_key, 'assignment.csv')

    print('OOS papers: ' + str(len(a.ignored_submissions)) + ': ' + str(a.ignored_submissions))
    print('OOS reviewers: ' + str(len(a.ignored_reviewers)) + ': ' + str(a.ignored_reviewers))
    print('# of inserted rev ' + str((a.num_rev + len(a.ignored_reviewers)) - len(similarity)))

    ################################################################################################################
    # evaluate assignment
    [max_a, mean_a, mean_a_w_pc, num_unused] = evaluate.calculate_max_mean_assignments(a.diverse_assignment,
                                                                                       a.rev_id_to_key,
                                                                                       a.ignored_reviewers)

    print("Max assignments per reviewer: " + str(max_a))
    print("Mean assignments per reviewer: " + str(mean_a))
    print("Mean assignments per reviewer (with original PC): " + str(mean_a_w_pc))
    print("# unused PC: " + str(num_unused))

    # fairness of assignment = sum similarity over reviewers assigned to a paper
    # fairness = minimum sum similarity across all papers, the higher the fairness, the better the assignment
    print('Fairness of the resulting assignment:' + str(evaluate.fairness(a.diverse_assignment, mle_est, a.similarity)))

    kl_diversity = evaluate.calc_kl_diversity(a.diverse_assignment, rev_id_to_key, textual_representation_of_reviewers)
    print('Avg KL Divergence: ' + str(kl_diversity))

    conflicts = evaluate.comp_violated_dependencies(a.diverse_assignment, a.dependant_revs)

    print('Percentage of assignments with conflicts of interest: ' + str(conflicts))

    print('Diversity of the resulting assignment: ' + str(a.best_diversity))

    # print assignments to file
    ct = 0
    with open('assignment_results.txt', 'w') as out_file:
        for assignment in a.diverse_assignment:
            out_file.write('________________________________________\nManuscript: ' + str(ct) + '\nRev IDs:\t')
            for i in range(demand):
                out_file.write(rev_id_to_key[a.diverse_assignment[assignment][i]] + '\t')
            out_file.write('\nNovel rev?\t')
            for i in range(demand):
                out_file.write(str(a.diverse_assignment[assignment][i] >= (initial_pc_size - num_unused)) + '\t')
            out_file.write('\nProf BG:\t')
            for i in range(demand):
                out_file.write(str(aca_ind[a.diverse_assignment[assignment][i]]) + '\t')
            out_file.write('\nLocation:\t')
            for i in range(demand):
                out_file.write(str(location[a.diverse_assignment[assignment][i]]) + '\t')
            out_file.write('\nSeniority:\t')
            for i in range(demand):
                out_file.write(str(seniority[a.diverse_assignment[assignment][i]]) + '\t')
            out_file.write('\n')
            out_file.flush()
            ct += 1
