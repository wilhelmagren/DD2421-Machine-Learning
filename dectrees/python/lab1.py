import dtree as dtree
import monkdata as md
import drawtree_qt5 as qt


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

    # Split the monk1 data into subsets according to the selected attribute using the function select and compute
    # the information gains for the nodes on the next level of the tree.
    splitted = dtree.select(m1, md.attributes[4], False)  # I don't know how to split it... what is the value?
    print(splitted)
    # the_tree = dtree.buildTree(m1, md.attributes)
    # qt.drawTree(the_tree)
    return


def main():
    monk1 = md.monk1
    monk2 = md.monk2
    monk3 = md.monk3
    #assignment1(monk1, monk2, monk3)
    #assignment3(monk1, monk2, monk3)
    assignment5(monk1, monk2, monk3)
    return


if __name__ == "__main__":
    main()
