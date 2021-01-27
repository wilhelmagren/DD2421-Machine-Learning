import dtree as dtree
import monkdata as md
import drawtree_qt5 as qt
import random
import matplotlib.pyplot as plt
import statistics


def assignment1(monk1, monk2, monk3):
    # Assignment 1: Import the files and use the entropy function to calculate the entropy of the monk datasets.
    # monk-1
    monk1_entropy = dtree.entropy(monk1)
    # monk-2
    monk2_entropy = dtree.entropy(monk2)
    # monk-3
    monk3_entropy = dtree.entropy(monk3)
    print(f"\n monk1 entropy: {monk1_entropy}\n monk2 entropy: {monk2_entropy}\n monk3_entropy: {monk3_entropy}")


def assignment3(monk1, monk2, monk3):
    # Assignment 2: done
    # Assignment 3: Use the function averageGain to calculate the expected information gain corresponding to each of
    #   the six attributes.
    print(f"\n\t monk1")
    for i in range(6):
        print(f"\n\t attribute {i + 1} i-gain: {dtree.averageGain(monk1, md.attributes[i])}")
    print(f"\n\t monk2")
    for i in range(6):
        print(f"\n\t attribute {i + 1} i-gain: {dtree.averageGain(monk2, md.attributes[i])}")
    print(f"\n\t monk3")
    for i in range(6):
        print(f"\n\t attribute {i + 1} i-gain: {dtree.averageGain(monk3, md.attributes[i])}")


def assignment5(m1, m2, m3):
    # Build the full decision trees for all three Monk datasets using buildTree/3.
    # Then, use the function check to measure the performance of the decision tree on both the training data
    # and the test datasets. Compute the train and test set errors for the three Monk datasets for the full trees.
    # Were your assumptions about the datasets correct? Explain the results you get for the training and test dataset.
    # tree = dtree.buildTree(m1, md.attributes)
    # print(dtree.check(tree, md.monk1test))
    # original_tree = dtree.buildTree(m1, md.attributes)
    # qt.drawTree(original_tree)
    # Split the monk1 data into subsets according to the selected attribute using the function select and compute
    # the information gains for the nodes on the next level of the tree.
    # for i in range(4):
    #    print(f"\n\t Tree number {i+1}")
    #    splitted = dtree.select(m1, md.attributes[4], i + 1)  # I don't know how to split it... what is the value?
    #    for i in range(6):
    #        print(f"\n\t Splitted tree, attribute {i + 1} i-gain: {dtree.averageGain(splitted, md.attributes[i])}")
    # Calculating the information gain?
    # for i in range(len(splitted)):
    #    print(splitted[i].attribute)

    # splitted = dtree.select(splitted, md.attributes[3], 2)
    # for i in range(6):
    #    print(f"\n\t Splitted tree, attribute {i + 1} i-gain: {dtree.averageGain(splitted, md.attributes[i])}")
    # print(dtree.mostCommon(splitted))
    # the_tree = dtree.buildTree(m1, md.attributes)
    tree_list = [dtree.buildTree(m1, md.attributes), dtree.buildTree(m2, md.attributes), dtree.buildTree(m3, md.attributes)]
    print(f"MONK-1 testing accuracy: [{1 - dtree.check(tree_list[0], md.monk1test)}]")
    print(f"MONK-2 testing accuracy: [{1 - dtree.check(tree_list[1], md.monk2test)}]")
    print(f"MONK-3 testing accuracy: [{1 - dtree.check(tree_list[2], md.monk3test)}]")
    return


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    break_point = int(len(ldata) * fraction)
    return ldata[:break_point], ldata[break_point:]


def pruning(m1, f):
    monk1train, monk1eval = partition(m1, f)
    tree = dtree.buildTree(monk1train, md.attributes)
    alternatives = dtree.allPruned(tree)
    while True:
        # print(f"\tThe current tree looks like: \n\t|-> {tree}")
        if len(alternatives) < 1:
            break
        alternatives = dtree.allPruned(tree)
        max_indx = -1
        max_acc = 0
        cur_acc = dtree.check(tree, monk1eval)
        # print(f"\tThe current tree acc is: {cur_acc}")
        for i, alt in enumerate(alternatives):
            # print(f"am I even here?")
            alt_acc = dtree.check(alt, monk1eval)
            # print(alt_acc)
            if alt_acc >= max_acc:
                max_acc = alt_acc
                max_indx = i
            # print(f"\t\tThe maximum acc found for pruned trees is: {max_acc}")
        if max_acc < cur_acc:
            # print(f"\t\tNo pruned tree has better accuracy than our current tree, returning the current tree...")
            break
        tree = alternatives[max_indx]
    return tree


def assignment7(m1, m3):
    frac_map = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    original_acc = []
    pruned_acc = []
    monk1_pruned_variance = []
    monk3_pruned_variance = []
    monk1_original_variance = []
    monk3_original_variance = []
    it = 1000
    for f in frac_map:
        mean_list_pruned = []
        mean_list_ori = []
        for i in range(it):
            # print(f"\tPruning on fraction {f}")
            monk1train, monk1eval = partition(m1, f)
            monk3train, monk3eval = partition(m3, f)
            m1tree, m3tree = dtree.buildTree(monk1train, md.attributes), dtree.buildTree(monk3train, md.attributes)
            acc_l = [dtree.check(m1tree, md.monk1test), dtree.check(m3tree, md.monk3test)]
            # print(f"\tThe original acc is: [{acc_l[0], acc_l[1]}]")
            prune1, prune3 = pruning(m1, f), pruning(m3, f)
            prune_l = [dtree.check(prune1, md.monk1test), dtree.check(prune3, md.monk3test)]
            # print(f"\tThe pruned acc is: [{prune_l[0], prune_l[1]}]")
            mean_list_ori.append(acc_l)
            mean_list_pruned.append(prune_l)
        pruned_avg_monk1 = 0
        pruned_avg_monk3 = 0
        ori_avg_monk1 = 0
        ori_avg_monk3 = 0
        tmp_1 = []
        tmp_3 = []
        for l in mean_list_pruned:
            tmp_1.append((1 - l[0])**2)
            tmp_3.append((1 - l[1])**2)
            pruned_avg_monk1 += l[0]
            pruned_avg_monk3 += l[1]
        monk1_pruned_variance.append(statistics.variance(tmp_1))
        monk3_pruned_variance.append(statistics.variance(tmp_3))
        tmp_1 = []
        tmp_3 = []
        for l in mean_list_ori:
            tmp_1.append((1 - l[0])**2)
            tmp_3.append((1 - l[1])**2)
            ori_avg_monk1 += l[0]
            ori_avg_monk3 += l[1]
        monk1_original_variance.append(statistics.variance(tmp_1))
        monk3_original_variance.append(statistics.variance(tmp_3))
        pruned_acc.append([(1 - pruned_avg_monk1/it)**2, (1 - pruned_avg_monk3/it)**2])
        original_acc.append([(1 - ori_avg_monk1/it)**2, (1 - ori_avg_monk3/it)**2])
    # print(f"\t\tThe accuracy lists are:\n\t\t|-> {original_acc, pruned_acc}")
    # Difference = PRUNED - ORIGINAL. Positive means pruned tree has better classification
    pruned_monk1, pruned_monk3, ori_monk1, ori_monk3 = [], [], [], []
    for l in pruned_acc:
        pruned_monk1.append(l[0])
        pruned_monk3.append(l[1])
    for l in original_acc:
        ori_monk1.append(l[0])
        ori_monk3.append(l[1])

    print(monk1_original_variance)
    plt.plot(frac_map, ori_monk1, color="blue", label="Original tree acc")
    plt.plot(frac_map, pruned_monk1, color="red", label="Pruned tree acc")
    plt.title("Original vs Pruned MONK-1 classification MSE")
    plt.xlabel("train/eval fraction")
    plt.ylabel(f"classification MSE, {it} iterations")
    plt.legend()
    plt.show()
    plt.plot(frac_map, ori_monk3, color="blue", label="Original tree acc")
    plt.plot(frac_map, pruned_monk3, color="red", label="Pruned tree acc")
    plt.title("Original vs Pruned MONK-3 classification MSE")
    plt.xlabel("train/eval fraction")
    plt.ylabel(f"classification MSE, {it} iterations")
    plt.legend()
    plt.show()
    plt.plot(frac_map, monk1_original_variance, color="blue", label="Original tree variance")
    plt.plot(frac_map, monk1_pruned_variance, color="red", label="Pruned tree variance")
    plt.title("Original vs Pruned MONK-1 classification MSE variance")
    plt.xlabel("train/eval fraction")
    plt.ylabel(f"MSE variance, {it} iterations")
    plt.legend()
    plt.show()
    plt.plot(frac_map, monk3_original_variance, color="blue", label="Original tree acc")
    plt.plot(frac_map, monk3_pruned_variance, color="red", label="Pruned tree acc")
    plt.title("Original vs Pruned MONK-3 classification MSE variance")
    plt.xlabel("train/eval fraction")
    plt.ylabel(f"MSE variance, {it} iterations")
    plt.legend()
    plt.show()
    return


def main():
    monk1 = md.monk1
    # monk2 = md.monk2
    monk3 = md.monk3
    # assignment1(monk1, monk2, monk3)
    # assignment3(monk1, monk2, monk3)
    # assignment5(monk1, monk2, monk3)
    assignment7(monk1, monk3)
    return


if __name__ == "__main__":
    main()
