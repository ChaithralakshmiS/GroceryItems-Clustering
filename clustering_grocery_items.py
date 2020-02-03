import sys
from fpgrowth import fp_growth
from Grocery_kmeans import k_means_execution,hierarchical_execution
if __name__ == "__main__":
    
    algo = str(sys.argv[1])
    if algo == "fpgrowth":
       fp_growth();
    elif algo == "kmeans":
        k_means_execution();
    elif algo == "hierarchical":
        hierarchical_execution();
    else:
       print("This algorithm is not implemented, or check your input string and try again.")
                     
        