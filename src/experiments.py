from metaheuristics import multi_start_local_search, iterated_local_search, genetic_local_search, multi_start_local_search_with_timelimit, iterated_local_search_with_timelimit, genetic_local_search_with_timelimit
from os import path
from graph import Graph
from constants import ROOT_DIR
from time import perf_counter
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

FM_PASSES = 10_000
RERUNS = 20
GRAPH = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))

def experiment_one():
    MLS_times, MLS_results = run_algorithm("MLS", GRAPH)
    ILS_times, ILS_results = run_algorithm("ILS", GRAPH, perturbation_factor=0.01)
    GLS_times, GLS_results = run_algorithm("GLS", GRAPH)

    MLS_time = np.mean(MLS_times)
    ILS_time = np.mean(ILS_times)
    GLS_time = np.mean(GLS_times)

    print(f"{MLS_time=} average time per pass")
    print(f"{ILS_time=} average time per pass")
    print(f"{GLS_time=} average time per pass")

    _, p_mls_ils = mannwhitneyu(MLS_results, ILS_results)
    _, p_mls_gls = mannwhitneyu(MLS_results, GLS_results)
    _, p_ils_gls = mannwhitneyu(ILS_results, GLS_results)

    print(f"{p_mls_ils=} p-value")
    print(f"{p_mls_gls=} p-value")
    print(f"{p_ils_gls=} p-value")

    fig, axs = plt.subplots(1, 3)
    ymin, ymax = min(min(MLS_results), min(ILS_results), min(GLS_results)), max(max(MLS_results), max(ILS_results), max(GLS_results))
    for ax in axs:
        ax.set_ylim([ymin, ymax])
    axs[0].boxplot(MLS_results)
    axs[0].set_title("MLS result")

    axs[1].boxplot(ILS_results)
    axs[1].set_title("ILS result")
    axs[2].boxplot(GLS_results)
    axs[2].set_title("GLS result")
    plt.savefig("experiment1.png")

    return MLS_results, ILS_results, GLS_results

def experiment_two():
    ILS_results; 

    # check if the worst and best performing ILS result for different perturbation differ significantly
    lowest, highest = np.min(ILS_results), np.max(ILS_results)
    _, p_perturbation = mannwhitneyu(lowest, highest)

def experiment_three():
    return 

def experiment_four(MLS_result, ILS_result):
    pvalues_mls_gls = []
    pvalues_ils_gls = []
    gls_global_results = []
    min_population_size = 50
    max_population_size = 1_250
    for population_size in range(min_population_size, max_population_size, 200):
        print(population_size)
        _, gls_results = run_algorithm("GLS", GRAPH, population_size=population_size)
        gls_global_results.append(gls_results)
        _, p_mls = mannwhitneyu(MLS_result, gls_results)
        _, p_ils = mannwhitneyu(ILS_result, gls_results)
        print(f"p value of mls vs gls with population size {population_size}: {p_mls}")
        print(f"p value of ils vs gls with population size {population_size}: {p_ils}")
        pvalues_mls_gls.append(p_mls)
        pvalues_ils_gls.append(p_ils)

    fig, axs = plt.subplots(1, 6)
    ymin, ymax = min([gls_global_results[i].any() for i in range(6)]), max([gls_global_results[i].any() for i in range(6)])
    for ax in axs:
        ax.set_ylim([ymin, ymax])
    axs[0].boxplot(gls_global_results[0])
    axs[0].set_title("population size 50")
    axs[1].boxplot(gls_global_results[1])
    axs[1].set_title("population size 250")
    axs[2].boxplot(gls_global_results[2])
    axs[2].set_title("population size 450")  
    axs[3].boxplot(gls_global_results[3])
    axs[3].set_title("population size 650")
    axs[4].boxplot(gls_global_results[4])
    axs[4].set_title("population size 850")
    axs[5].boxplot(gls_global_results[5])
    axs[5].set_title("population size 1050")

def experiment_four_two():
    start = perf_counter()
    run_algorithm("MLS", GRAPH)
    end = perf_counter()
    one_mls_runtime = (end - start) / 20

    MLS_results = np.zeros(shape=25)
    ILS_results = np.zeros(shape=25)
    GLS_results = np.zeros(shape=25)

    for i in range(25):
        _, mls_best_cutsize = multi_start_local_search_with_timelimit(GRAPH, time_limit=one_mls_runtime)
        MLS_results[i] = mls_best_cutsize

    for i in range(25):
        _, ils_best_cutsize = iterated_local_search_with_timelimit(GRAPH, time_limit=one_mls_runtime, perturbation_factor=0.01)
        ILS_results[i] = ils_best_cutsize
    
    for i in range(25):
        _, gls_best_cutsize = genetic_local_search_with_timelimit(GRAPH, time_limit=one_mls_runtime, population_size=50)
        GLS_results[i] = gls_best_cutsize


    fig, axs = plt.subplots(1, 3)
    ymin, ymax = min(min(MLS_results), min(ILS_results), min(GLS_results)), max(max(MLS_results), max(ILS_results), max(GLS_results))
    for ax in axs:
        ax.set_ylim([ymin, ymax])
    axs[0].boxplot(MLS_results)
    axs[0].set_title("MLS result")

    axs[1].boxplot(ILS_results)
    axs[1].set_title("ILS result")
    axs[2].boxplot(GLS_results)
    axs[2].set_title("GLS result")
    plt.savefig("experiment4b.png")

def run_algorithm(algorithm, graph, perturbation_factor=0, population_size=50):
    if algorithm == "MLS":
        MLS_times = np.zeros(shape=RERUNS)      
        MLS_results = np.zeros(shape=RERUNS)
        for i in range(20):
            start = perf_counter()
            _, result_MLS_cutsize = multi_start_local_search(graph, FM_PASSES)
            end = perf_counter()
            time = end - start
            MLS_times[i] = time
            MLS_results[i] = result_MLS_cutsize
        return MLS_times, MLS_results

    elif algorithm == "ILS":
        ILS_times = np.zeros(shape=RERUNS)
        ILS_results = np.zeros(shape=RERUNS)
        for i in range(20):
            start = perf_counter()
            _, result_ILS_cutsize = iterated_local_search(graph, FM_PASSES, perturbation_factor=perturbation_factor)
            end = perf_counter()
            time = end - start
            ILS_times[i] = time
            ILS_results[i] = result_ILS_cutsize
        return ILS_times, ILS_results

    elif algorithm == "GLS":
        GLS_times = np.zeros(shape=RERUNS)
        GLS_results = np.zeros(shape=RERUNS)
        for i in range(20):
            start = perf_counter()
            _, result_GLS_cutsize = genetic_local_search(graph, population_size=population_size, max_iterations=FM_PASSES)
            end = perf_counter()
            time = end - start
            GLS_times[i] = time
            GLS_results[i] = result_GLS_cutsize
        return GLS_times, GLS_results

MLS_results, ILS_results, GLS_results = experiment_one()
experiment_four_two()
experiment_four(MLS_results, ILS_results)


