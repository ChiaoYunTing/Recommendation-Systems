
import os
import numpy as np

def read_rating(path, num_users, num_items, num_total_ratings, a, b, train_ratio):
    # Construct the full path to the ratings file
    file_path = os.path.join(path, "ratings.dat")
    
    # Initialize sets to store users and items in train and test sets
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    # Initialize matrices for ratings, masks, and confidence
    R = np.zeros((num_users, num_items))  # Full rating matrix
    mask_R = np.zeros((num_users, num_items))  # Mask for existing ratings
    C = np.ones((num_users, num_items)) * b  # Confidence matrix
    train_R = np.zeros((num_users, num_items))  # Training ratings
    test_R = np.zeros((num_users, num_items))  # Test ratings
    train_mask_R = np.zeros((num_users, num_items))  # Mask for training ratings
    test_mask_R = np.zeros((num_users, num_items))  # Mask for test ratings
    
    # Randomly split data into train and test sets
    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[0:int(num_total_ratings*train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings*train_ratio):]
    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)

    # Read all lines from the ratings file
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
    
    # Process all ratings
    for line in lines:
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        R[user_idx, item_idx] = int(rating)
        mask_R[user_idx, item_idx] = 1
        C[user_idx, item_idx] = a
    
    # Process training data
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_R[user_idx, item_idx] = int(rating)
        train_mask_R[user_idx, item_idx] = 1
        user_train_set.add(user_idx)
        item_train_set.add(item_idx)
    
    # Process test data
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_R[user_idx, item_idx] = int(rating)
        test_mask_R[user_idx, item_idx] = 1
        user_test_set.add(user_idx)
        item_test_set.add(item_idx)
    
    # Return all processed data
    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
           user_train_set, item_train_set, user_test_set, item_test_set
