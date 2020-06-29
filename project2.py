import pandas as pd
import numpy as np
import math
import random

class TrainUser:
	def __init__(self, userid, ratings):
		self.userid = userid
		self.ratings = ratings

	def __repr__(self):
		return str(self.userid) + " " + str(self.ratings)

class TestUser:
	def __init__(self, userid, ratings, predictions):
		self.userid = userid
		self.ratings = ratings
		self.predictions = predictions

	def __repr__(self):
		return str(self.userid) + " " + str(self.ratings) + " " + str(self.predictions)

#calculate the average of the user's ratings
def user_average_rating(ratings, movieid):
	sum = 0
	count = 0
	for key in ratings:
		if ((key != movieid) and (ratings[key] != 0)):
			sum += ratings[key]
			count += 1
	return sum / count

def user_cosine_similarity(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for train_user in train_users:
		#if the train user rated the movie
		if (train_user.ratings[movieid] != 0):
			vector1 = []
			vector2 = []
			for key in test_user.ratings:
				#if the test user and the train user rated the movie
				if (train_user.ratings[key] != 0):
					vector1.append(train_user.ratings[key])
					vector2.append(test_user.ratings[key])
			#if the lengths of the vectors are 1, the cosine similarity is 1
			if ((len(vector1) > 1) and (len(vector2) > 1)):
				#calculate the cosine similarity
				weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
				#the weight is the cosine similarity
				numerator += weight * train_user.ratings[movieid]
				denominator += weight
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return numerator / denominator
	else:
		return user_average_rating(train_user.ratings, movieid)

#calculate the global mean
def global_mean(train_users):
	sum = 0
	count = 0
	for train_user in train_users:
		for key in train_user.ratings:
			if (train_user.ratings[key] != 0):
				sum += train_user.ratings[key]
				count += 1
	return sum / count

#calculate the user mean
def user_mean(ratings, movieid):
	sum = 0
	count = 0
	for key in ratings:
		if ((key != movieid) and (ratings[key] != 0)):
			sum += ratings[key]
			count += 1
	return sum / count

#calculate the shrunk mean
def shrunk_mean(ratings, movieid, user_mean, global_mean):
	count = 0
	for key in ratings:
		if ((key != movieid) and (ratings[key] != 0)):
			count += 1
	return ((count / (1 + count)) * user_mean) + ((1 / (1 + count)) * global_mean)

def user_pearson_correlation_smoothing(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for train_user in train_users:
		#if the train user rated the movie
		if (train_user.ratings[movieid] != 0):
			vector1 = []
			vector2 = []
			for key in test_user.ratings:
				#if the test user and the train user rated the movie
				if (train_user.ratings[key] != 0):
					vector1.append(train_user.ratings[key])
					vector2.append(test_user.ratings[key])
			if ((len(vector1) != 0) and (len(vector2) != 0)):
				#calculate the Pearson correlation
				vector1 = np.asarray(vector1, dtype = float)
				vector2 = np.asarray(vector2, dtype = float)
				train_users_global_mean = 3.5824010999312543
				train_user_user_mean = user_mean(train_user.ratings, movieid)
				test_user_user_mean = user_mean(test_user.ratings, movieid)
				train_user_average_rating = shrunk_mean(train_user.ratings, movieid, train_user_user_mean, train_users_global_mean)
				test_user_average_rating = shrunk_mean(test_user.ratings, movieid, test_user_user_mean, train_users_global_mean)
				vector1 -= train_user_average_rating
				vector2 -= test_user_average_rating
				weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
				if (not math.isnan(weight)):
					#the weight is the Pearson correlation
					numerator += weight * (train_user.ratings[movieid] - train_user_average_rating)
					denominator += abs(weight)
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return test_user_average_rating + (numerator / denominator)
	else:
		return user_average_rating(train_user.ratings, movieid)

def user_pearson_correlation(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for train_user in train_users:
		#if the train user rated the movie
		if (train_user.ratings[movieid] != 0):
			vector1 = []
			vector2 = []
			for key in test_user.ratings:
				#if the test user and the train user rated the movie
				if (train_user.ratings[key] != 0):
					vector1.append(train_user.ratings[key])
					vector2.append(test_user.ratings[key])
			if ((len(vector1) != 0) and (len(vector2) != 0)):
				#calculate the Pearson correlation
				vector1 = np.asarray(vector1, dtype = float)
				vector2 = np.asarray(vector2, dtype = float)
				train_user_average_rating = user_average_rating(train_user.ratings, movieid)
				test_user_average_rating = user_average_rating(test_user.ratings, movieid)
				vector1 -= train_user_average_rating
				vector2 -= test_user_average_rating
				weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
				if (not math.isnan(weight)):
					#the weight is the Pearson correlation
					numerator += weight * (train_user.ratings[movieid] - train_user_average_rating)
					denominator += abs(weight)
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return test_user_average_rating + (numerator / denominator)
	else:
		return user_average_rating(train_user.ratings, movieid)

#calculate IUF
def iuf(train_users, movieid):
	count = 0
	for train_user in train_users:
		if (train_user.ratings[movieid] != 0):
			count += 1
	return math.log2(len(train_users) / count)

def user_pearson_correlation_iuf(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for train_user in train_users:
		#if the train user rated the movie
		if (train_user.ratings[movieid] != 0):
			vector1 = []
			vector2 = []
			iuf_weights = []
			for key in test_user.ratings:
				if (train_user.ratings[key] != 0):
					#if the test user and the train user rated the movie
					vector1.append(train_user.ratings[key])
					vector2.append(test_user.ratings[key])
					iuf_weights.append(iuf(train_users, key))
			if ((len(vector1) != 0) and (len(vector2) != 0)):
				#calculate the Pearson correlation
				vector1 = np.asarray(vector1, dtype = float)
				vector2 = np.asarray(vector2, dtype = float)
				train_user_average_rating = user_average_rating(train_user.ratings, movieid)
				test_user_average_rating = user_average_rating(test_user.ratings, movieid)
				vector1 -= train_user_average_rating
				vector2 -= test_user_average_rating
				#the ratings are the original ratings multiplied by IUF
				vector1 *= iuf_weights
				vector2 *= iuf_weights
				weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
				if (not math.isnan(weight)):
					#the weight is the Pearson correlation
					numerator += weight * (train_user.ratings[movieid] - train_user_average_rating)
					denominator += abs(weight)
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return test_user_average_rating + (numerator / denominator)
	else:
		return user_average_rating(train_user.ratings, movieid)

def user_pearson_correlation_case(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for train_user in train_users:
		#if the train user rated the movie
		if (train_user.ratings[movieid] != 0):
			vector1 = []
			vector2 = []
			for key in test_user.ratings:
				#if the test user and the train user rated the movie
				if (train_user.ratings[key] != 0):
					vector1.append(train_user.ratings[key])
					vector2.append(test_user.ratings[key])
			if ((len(vector1) != 0) and (len(vector2) != 0)):
				#calculate the Pearson correlation
				vector1 = np.asarray(vector1, dtype = float)
				vector2 = np.asarray(vector2, dtype = float)
				train_user_average_rating = user_average_rating(train_user.ratings, movieid)
				test_user_average_rating = user_average_rating(test_user.ratings, movieid)
				vector1 -= train_user_average_rating
				vector2 -= test_user_average_rating
				weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
				if (not math.isnan(weight)):
					weight = weight * (abs(weight) ** 1.5)
					#the weight is the Pearson correlation multiplied by the case amplification
					numerator += weight * (train_user.ratings[movieid] - train_user_average_rating)
					denominator += abs(weight)
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return test_user_average_rating + (numerator / denominator)
	else:
		return user_average_rating(train_user.ratings, movieid)

def item_cosine_similarity(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for key in test_user.ratings:
		vector1 = []
		vector2 = []
		for train_user in train_users:
			#if the test user and the train user rated the movie
			if ((train_user.ratings[key] != 0) and (train_user.ratings[movieid] != 0)):
				vector1.append(train_user.ratings[key])
				vector2.append(train_user.ratings[movieid])
		#if the lengths of the vectors are 1, the cosine similarity is 1
		if ((len(vector1) > 1) and (len(vector2) > 1)):
			#calculate the cosine similarity
			weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
			#the weight is the cosine similarity
			numerator += weight * test_user.ratings[key]
			denominator += weight
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return numerator / denominator
	else:
		return user_average_rating(train_user.ratings, movieid)

def ensemble(train_users, test_user, movieid):
	#the predicted rating is the average rating of the methods
	ratings = []
	ratings.append(user_cosine_similarity(train_users, test_user, movieid))
	ratings.append(user_pearson_correlation_smoothing(train_users, test_user, movieid))
	return sum(ratings) / len(ratings)

#calculate the output of the sigmoid function
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

#calculate the output of the derivative of the sigmoid function
def derivative(x):
	return x * (1 - x)

#optimize the weights by training the neural network
def learn(train_user, vector1, vector2, movieid, weight):
	vector1 = np.array(vector1)
	vector2 = np.array(vector2)
	#normalize the ratings
	if (np.std(vector1) == 0):
		vector1 = [.5] * len(vector1)
	else:
		vector1 = (vector1 - np.mean(vector1)) / np.std(vector1)
	#initialize the weights to the cosine similarity
	weights = [weight] * len(vector1)
	weights = np.array(weights)
	#adjust the weights to minimize the cost function
	for i in range(100):
		value = sigmoid(np.sum(vector1 * weights))
		error = train_user.ratings[movieid] - value
		weights = weights * (error * derivative(value))
	return (sigmoid(np.sum(vector2 * weights)) * np.std(vector2)) + np.mean(vector2)

def deep_learning(train_users, test_user, movieid):
	numerator = 0
	denominator = 0
	for train_user in train_users:
		#if the train user rated the movie
		if (train_user.ratings[movieid] != 0):
			vector1 = []
			vector2 = []
			for key in test_user.ratings:
				#if the test user and the train user rated the movie
				if (train_user.ratings[key] != 0):
					vector1.append(train_user.ratings[key])
					vector2.append(test_user.ratings[key])
			#if the lengths of the vectors are 1, the cosine similarity is 1
			if ((len(vector1) > len(test_user.ratings) / 2) and (len(vector2) > len(test_user.ratings) / 2)):
				#calculate the cosine similarity
				weight = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
				#the weights are optimized by training the neural network
				rating = learn(train_user, vector1, vector2, movieid, weight)
				numerator += rating
				denominator += 1
	#if the movie is not rated, the predicted rating is the average of the user's ratings
	if (denominator != 0):
		return numerator / denominator
	else:
		return user_average_rating(train_user.ratings, movieid)

def predict_ratings(train, test, result, function):
	train_df = pd.read_csv(train, sep = "	", header = None)
	train_users = []
	for i in range(len(train_df)):
		ratings = {}
		for j in range(len(train_df.values[i])):
			ratings[j + 1] = train_df.values[i][j]
		train_user = TrainUser(i + 1, ratings)
		train_users.append(train_user)
	test_df = pd.read_csv(test, sep = " ", names = ["user", "movie", "rating"])
	test_users = []
	users = test_df["user"].unique()
	for user in users:
		group = test_df.groupby("user").get_group(user)
		ratings = {}
		predictions = []
		for i in range(len(group)):
			if (group.values[i][2] == 0):
				predictions.append(group.values[i][1])
			else:
				ratings[group.values[i][1]] = group.values[i][2]
		test_user = TestUser(user, ratings, predictions)
		test_users.append(test_user)
	lines = []
	for test_user in test_users:
		for movieid in test_user.predictions:
			rating = int(round(function(train_users, test_user, movieid)))
			if (rating < 1):
				rating = 1
			if (rating > 5):
				rating = 5
			lines.append(str(test_user.userid) + " " + str(movieid) + " " + str(rating) + "\n")
	file = open(result, "w")
	file.writelines(lines)
	file.close()

# predict_ratings("train.txt", "test5.txt", "result5.txt", user_cosine_similarity)
# predict_ratings("train.txt", "test10.txt", "result10.txt", user_cosine_similarity)
# predict_ratings("train.txt", "test20.txt", "result20.txt", user_cosine_similarity)
# predict_ratings("train.txt", "test5.txt", "result5.txt", user_pearson_correlation)
# predict_ratings("train.txt", "test10.txt", "result10.txt", user_pearson_correlation)
# predict_ratings("train.txt", "test20.txt", "result20.txt", user_pearson_correlation)
# predict_ratings("train.txt", "test5.txt", "result5.txt", user_pearson_correlation_iuf)
# predict_ratings("train.txt", "test10.txt", "result10.txt", user_pearson_correlation_iuf)
# predict_ratings("train.txt", "test20.txt", "result20.txt", user_pearson_correlation_iuf)
# predict_ratings("train.txt", "test5.txt", "result5.txt", user_pearson_correlation_case)
# predict_ratings("train.txt", "test10.txt", "result10.txt", user_pearson_correlation_case)
# predict_ratings("train.txt", "test20.txt", "result20.txt", user_pearson_correlation_case)
# predict_ratings("train.txt", "test5.txt", "result5.txt", item_cosine_similarity)
# predict_ratings("train.txt", "test10.txt", "result10.txt", item_cosine_similarity)
# predict_ratings("train.txt", "test20.txt", "result20.txt", item_cosine_similarity)
# predict_ratings("train.txt", "test5.txt", "result5.txt", ensemble)
# predict_ratings("train.txt", "test10.txt", "result10.txt", ensemble)
# predict_ratings("train.txt", "test20.txt", "result20.txt", ensemble)
# predict_ratings("train.txt", "test5.txt", "result5.txt", deep_learning)
# predict_ratings("train.txt", "test10.txt", "result10.txt", deep_learning)
# predict_ratings("train.txt", "test20.txt", "result20.txt", deep_learning)

def MAE(actual_ratings, predicted_ratings):
	sum = 0
	for key in predicted_ratings:
		sum += abs((actual_ratings[key] - predicted_ratings[key]))
	return sum / len(actual_ratings)

def cross_validation(train, function):
	train_df = pd.read_csv(train, sep = "	", header = None)
	train_users = []
	test_users = []
	actual_ratings = {}
	predicted_ratings = {}
	for i in range(len(train_df)):
		if (i < len(train_df) * .8):
			ratings = {}
			for j in range(len(train_df.values[i])):
				ratings[j + 1] = train_df.values[i][j]
			train_user = TrainUser(i + 1, ratings)
			train_users.append(train_user)
		else:
			number = random.randint(0, 1)
			ratings = {}
			predictions = []
			for j in range(len(train_df.values[i])):
				if (train_df.values[i][j] != 0):
					if (number == 0):
						ratings[j + 1] = train_df.values[i][j]
					else:
						predictions.append(j + 1)
						actual_ratings[j + 1] = train_df.values[i][j]
			test_user = TestUser(i + 1, ratings, predictions)
			test_users.append(test_user)
	sum = 0
	for test_user in test_users:
		for movieid in test_user.predictions:
			rating = round(function(train_users, test_user, movieid))
			if (rating < 1):
				rating = 1
			if (rating > 5):
				rating = 5
			predicted_ratings[movieid] = rating
		sum += MAE(actual_ratings, predicted_ratings)
	return sum / len(test_users)

# print(cross_validation("train.txt", user_cosine_similarity))
# print(cross_validation("train.txt", user_pearson_correlation))
# print(cross_validation("train.txt", user_pearson_correlation_iuf))
# print(cross_validation("train.txt", user_pearson_correlation_case))
# print(cross_validation("train.txt", item_cosine_similarity))
# print(cross_validation("train.txt", ensemble))
# print(cross_validation("train.txt", deep_learning))