from math import floor # BalancedClusters.get_smaller_than_average_cluster_names
import pandas as pd # BalancedClusters.plot_clusters
import seaborn as sns # BalancedClusters.plot_clusters
import matplotlib.pyplot as plt # BalancedClusters.plot_clusters


class ElementMovers:
    move_to_smallest = 'smallest'
    move_to_optimal_smaller = 'optimal'

class StaticMethods:
    '''
    Methods that are used by BalancedClusters, but
    not unique to it.
    '''
    @staticmethod
    def argmin(D):
        '''
        Argmin for a dictionary, D.
        '''
        return min(D, key=D.get)

    @staticmethod
    def dictionary_check_equal(d1, d2):
        '''
        Checks whether two cluster dictionaries share
        the exact same elements.
        '''
        if d1.keys() != d2.keys(): return False
        num_equal = sum([1 if(np.array_equal(d1[k], d2[k])) else 0 for k in d1.keys()])
        return True if num_equal == len(d1.keys()) else False

    @staticmethod
    def distance_metric( a_coords, b_coords):
        '''
        This is flexible, but right now it's Euclidean
        distance. It could alternatively be absolute distance,
        or absolute distance cubed, or whatever.'''
        difference_array = a_coords - b_coords
        return np.linalg.norm(difference_array)

class BalancedClusters:

    def __init__(self, cluster_dict_inp, element_mover):
        '''
        Summary:
            Constructor, setting class variables
        Parameters:
            element_mover: a function that takes in the parameters below and then edits
                           the cluster dictionary taken as input
        '''
        self.cluster_dict = cluster_dict_inp.copy() # copy once, so as to not edit
        if element_mover == ElementMovers.move_to_smallest:
            self.element_mover = self.move_element_to_smallest_cluster
        elif element_mover == ElementMovers.move_to_optimal_smaller:
            self.element_mover = self.move_element_to_optimal_smaller_cluster

    def get_centroid(self, cluster_elements):
        return cluster_elements.mean(axis=0)

    def get_centroids(self): return {k : self.get_centroid(self.cluster_dict[k]) for k in self.cluster_dict.keys()}

    def get_group_num_elements(self, group_name):
        return self.cluster_dict[group_name].shape[0]

    def get_all_group_sizes(self):
        return {group_name : self.get_group_num_elements(group_name) for group_name in self.cluster_dict.keys()}

    def get_group_with_fewest_elements(self):
        '''
        Given a dictionary, mapping from group name
        to a numpy array of group element coordinates,
        return the name (AKA key) of the group that has
        the fewest number of elements. Ties are not handled
        arbitrarily.
        '''
        group_names = list(self.cluster_dict.keys()) # create indexable list
        num_elements_list = [ self.get_group_num_elements(group_name) for group_name in group_names ] # find number of elements for each key
        index_of_min = np.argmin(num_elements_list) # find index of min for the key
        return group_names[index_of_min] # return the key at the index where the min occurred

    def get_smaller_than_average_cluster_names(self):
        '''
        Returns names of clusters that need to increase their
        number of elements in order to balance the clusters
        over all. This means any cluster with fewer elements
        than the floor of the average will be returned.
        '''
        total_num_elements = sum([self.get_group_num_elements(group_name) for group_name in self.cluster_dict.keys()])
        avg_group_size = total_num_elements / len(self.cluster_dict.keys())
        group_sizes = self.get_all_group_sizes()
        smaller_than_avg_cluster_names = []
        for group_name in self.cluster_dict.keys():
            if ( self.get_group_num_elements(group_name) <= floor(avg_group_size) ):
                smaller_than_avg_cluster_names += [group_name]
        return smaller_than_avg_cluster_names

    def plot_clusters(self):
        '''
        The gist of this is to create a
        dataframe with columns being
        the x, y, and group name, and
        then plotting this in seaborn
        with the hue being the group name.
        '''
        def get_total_num_elements():
            '''
            Just finds how many elements there
            are total in the cluster_dict (not
            how many numbers, but how many
            elements; an element may have
            multiple dimensions)'''
            total = 0
            for group_name in self.cluster_dict.keys():
                total += self.get_group_num_elements(group_name)
            return total
        
        num_elements = get_total_num_elements()
        all_elements = [ [0 for j in range(3)] for i in range(num_elements) ]
        row_counter = 0
        for group_name in self.cluster_dict.keys():
            cluster = self.cluster_dict[group_name]
            for element in cluster:
                all_elements[row_counter] = [group_name] + element.tolist()
                row_counter += 1
        all_elements_df = pd.DataFrame(all_elements, columns=['group_name', 'x', 'y'])
        sns.scatterplot(x='x', y='y', hue='group_name', data=all_elements_df)
        plt.show()

    
    # ## Important Methods

    def get_distances_from_small_group_centroid(self, small_group_name):
        '''
        Goes and looks at all specified groups in a given
        cluster dictionary. It reports the distance from
        each individual point to the reference element.
        
        small_group_name: name of the group that should be treated as the
                          small group. The small group will have distances
                          of larger groups calculated against it.
        return: 
            dictionary {
                distance : dictionary{
                'group_name': <string group name>, 
                'element': <np.array element values>
                } 
            }
            
        '''
        
        other_group_names = set(self.cluster_dict.keys())
        other_group_names.remove(small_group_name)
        
        centroids = self.get_centroids()
        small_group_centroid = centroids[small_group_name]
        
        small_group_num_elements = self.get_group_num_elements(small_group_name)
        
        distances_from_small_group_centroid = {} 
        for group_name in other_group_names:
            group_num_elements = self.get_group_num_elements(group_name)
            # The rule is that we can only take elements from a cluster that currently
            # has at least 2 more elements than the cluster with the least number of elements
            if (group_num_elements - small_group_num_elements) >= 2:
                group_elements = self.cluster_dict[group_name]
                # This could be sped up with list comprehension, but not implementing at this stage
                # so as to preserve clarity
                for element in group_elements:
                    distance_to_small_group_centroid = StaticMethods.distance_metric(
                        small_group_centroid, element)
                    distances_from_small_group_centroid[distance_to_small_group_centroid] = {
                        'group_name':group_name, 'element':element
                    }
        return distances_from_small_group_centroid

    def move_element(self, small_group_name, large_group_name, moving_element, verbose):
        # cluster_dict = cluster_dict_inp.copy()
        
        # Get the index of the row that should be taken out of the old cluster
        index_of_deletion = np.argwhere(self.cluster_dict[large_group_name]==moving_element)[0, 0]
        # Move row out of old cluster
        self.cluster_dict[large_group_name] = np.delete(self.cluster_dict[large_group_name], index_of_deletion, axis=0)
        # Move row to new, small cluster
        self.cluster_dict[small_group_name] = np.append( self.cluster_dict[small_group_name], [moving_element], axis=0 )

        if verbose.lower() in ('text', 'all'): print(f'Moving {moving_element} from cluster '
                                                         + f'{large_group_name}'
                                                          + f' to cluster {small_group_name}')

    def move_element_to_smallest_cluster(self, verbose):
        '''
        Description:
            This method takes in clusters and then moves an element from a list with more
            elements to the list with the fewest elements. Iterating on this stepper 
            guarantees making balanced clusters (i.e. it's impossible to get stuck in an
            infinite loop of moving elements in circles).
        
        Parameters:
            cluster_dict_inp: a dictionary that maps from cluster names to numpy arrays;
                                the dimension of the array is the dimension of a sample
            verbose: string, specifying the mode of verbosity; if it is 'all', then
                     both text and plots will be displayed; if 'plot', then only plots
                     will be displayed, not text; if 'text', then only text will be 
                     displayed, not plots.
        Return:
            Returns a cluster dictionary with one of the elements moved from a larger group to the smallest
            group.
        '''
        
        if verbose.lower() in ('text', 'all'): print(f'Group sizes: {self.get_all_group_sizes()}')
        small_group_name = self.get_group_with_fewest_elements()

        # See how far the small group is from elements in larger groups
        distances_from_small_group_centroid = self.get_distances_from_small_group_centroid(small_group_name)
        
        if distances_from_small_group_centroid:
            # If the method returned an empty dictionary of possible distances,
            # this implies that there are no further changes to make, and thus the
            # function is finished running.

            # Now find the smallest centroid and move it from its current cluster to the smallest cluster
            distances_from_small_group_centroid_list = distances_from_small_group_centroid.keys()
            shortest_dist_from_small_centroid = min(distances_from_small_group_centroid_list)
            moving_element_information = distances_from_small_group_centroid[shortest_dist_from_small_centroid]

            moving_element_previous_group_name = moving_element_information['group_name']
            moving_element = moving_element_information['element']

            self.move_element(small_group_name, moving_element_previous_group_name,
                                       moving_element, verbose)
            
    def get_smallest_dist_move(self, distance_dict):
        '''
        input: dictionary{ key = <small group name>, 
                          value = <dictionary {key = <distance>,
                                               value = dictionary { key = 'group_name' : value = <group name>
                                                                    key = 'element' : value = <numpy array of element value>                                                           
                                                                  }
                                               }
                         }
        returns: tuple of (<small group name>, <large group name>, <element>)
        '''
        # Getting the smallest possible move
        small_group_names = distance_dict.keys()
        
        small_group_name_to_smallest_distance = {}
        for small_group_name in small_group_names:
            if distance_dict[small_group_name]:
                small_group_name_to_smallest_distance[small_group_name] = min(
                    distance_dict[small_group_name].keys())
        
        if small_group_name_to_smallest_distance.keys():                                                                  
            # Getting name of small group that will receive element (optimal group to receive element)
            smallest_move_small_group_name = StaticMethods.argmin(small_group_name_to_smallest_distance)

            # Getting information of the group that the element is getting moved from
            smallest_move_dist = small_group_name_to_smallest_distance[smallest_move_small_group_name]
            element_to_move_info = distance_dict[smallest_move_small_group_name][smallest_move_dist]
            element_previous_group_name = element_to_move_info['group_name']
            element = element_to_move_info['element']
            return (smallest_move_small_group_name, element_previous_group_name, element)
        else:
            return (None, None, None)

    def move_element_to_optimal_smaller_cluster(self, verbose):
        '''
        Description:
            Moves an element from to one of the clusters with fewer elements than average, by
            taking an element from one of the clusters with at least two more elements. Very
            similar to move_element_to_smallest(), but instead this method does not have
            to move an element to the smallest cluster; it just needs to move an element to
            a cluster that has fewer elements than average. This is the minimal constraint
            for making monotonic progress toward balanced classes (that I've found, at least).
        Parameters:
            cluster_dict_inp:  a dictionary that maps from cluster names to numpy arrays;
                                the dimension of the arrays is the dimension of a sample
            verbose: string specifying the mode of verbosity; if it is 'all', then
                     both text and plots will be displayed; if 'plot', then only plots
                     will be displayed, not text; if 'text', then only text will be 
                     displayed, not plots.
        Return:
            Returns a new cluster dictionary.
        '''
        
        if verbose.lower() in ('text', 'all'): print(f'Group sizes: {get_all_group_sizes()}')
        
        # Get names of clusters that could receive more elements
        small_group_names = self.get_smaller_than_average_cluster_names()
        if verbose.lower() in ('all', 'text'): print(f'small_group_names: {small_group_names}')
        small_group_name_to_moving_distance_dictionary = {group_name : 
                                                          self.get_distances_from_small_group_centroid(group_name)
                                                          for group_name in small_group_names}
        # Get the information of the move that has the smallest distance metric
        small_group_name, large_group_name, moving_element = self.get_smallest_dist_move(
            small_group_name_to_moving_distance_dictionary)

        if ((small_group_name != None) and (large_group_name != None) and (moving_element.size != 0)):
            self.move_element(small_group_name, large_group_name, moving_element, verbose)
        
    def balance_clusters(self, max_iterations=100, verbose='text'):
        '''
        Description:
            This method takes in clusters and then balances all of the clusters so
            that they have the same number of elements (or are within one of each other).
        
        Parameters:
            max_iterations: the maximum number of iterations before the algorithm stops;
                            this is a fail-safe from indefinite looping (although this
                            should not be an issue if the code is correct)
            verbose: string, specifying the mode of verbosity; if it is 'all', then
                     both text and plots will be displayed; if 'plot', then only plots
                     will be displayed, not text; if 'text', then only text will be 
                     displayed, not plots.
        Return:
            a dictionary of the exact same form as cluster_dict_inp, except with balanced
            clusters
        '''
        # cluster_dict = cluster_dict_inp.copy() # copy, so as not to edit original

        counter = 0 
        while counter < max_iterations:
            cluster_dict_old = self.cluster_dict.copy()
            self.element_mover(verbose) # element_mover should not return, but instead edit
            if StaticMethods.dictionary_check_equal(cluster_dict_old, self.cluster_dict): 
                if verbose.lower() in ('text', 'all'): print('There are no more small groups.')
                return self.cluster_dict # no changes were made to cluster dictionary, so return it
            else:
                # Plotting only works for 2 dimensional data. This checks if the data is 2D
                is2d = len(self.cluster_dict[list(self.cluster_dict.keys())[0]][0]) == 2
                if verbose.lower() in ('plot', 'all') and is2d: plot_clusters(self.cluster_dict)
                counter += 1
        return self.cluster_dict

# Testing BalanceClusters
import numpy as np
def make_cluster(approx_centroid, noise_max, num_elements):
    '''
    Generates a cluster-ish area around the
    approx_centroid coordinates.
    
    Returns an 
    '''
    cluster = []
    for i in range(num_elements):
        noise = np.random.uniform(low=-noise_max, high=noise_max, size=len(approx_centroid))
        coordinates = np.array(approx_centroid) + noise
        cluster += [coordinates]
    return np.array(cluster)

approx_centroids = [[0, 0], [10, 0], [0, 10], [10, 10]]
clusters = {
    'A' : make_cluster([0, 0], 2, 100), # based around (0, 0)
    'B' : make_cluster([10, 0], 2, 120), # based around (10, 0)
    'C' : make_cluster([0, 10], 2, 85), # based around (0, 10)
    'D' : make_cluster([10, 10], 2, 110) # based around (10, 10)
}

from time import time
start = time()
simpleclusters = BalancedClusters(clusters, 'smallest')
balanced_clusters_move_to_smallest = simpleclusters.balance_clusters(max_iterations=1000, verbose='none')
end = time()
print(f"Finished simpler way in {end-start} seconds.")
simpleclusters.plot_clusters()
print(simpleclusters.get_all_group_sizes())


optimalclusters = BalancedClusters(clusters, 'optimal')
start = time()
balanced_clusters_move_to_optimal = optimalclusters.balance_clusters(verbose='none')
end = time()
print(f"Finished more complicated way in {end-start} seconds.")
print(optimalclusters.get_all_group_sizes())
optimalclusters.plot_clusters()