

# -*- coding: utf-8 -*-
"""
A program to carry out Kmeans clustering 
on data from Visa using methods learned in
"Data Smart: Using Data Science to Transform Information into Insight"

Requires csv input file ticket.csv 
"""

#make more similar to Python 3
from __future__ import print_function, division, absolute_import, unicode_literals

#other stuff we need to import
import csv
import numpy as np
from sklearn.cluster import KMeans

#beginning of main program

#read in ticket.csv
csvf = open('ticket.csv','rU')
rows = csv.reader(csvf)
ticketmatrix = [row for row in rows]
csvf.close()



#first row of each spreadsheet is column headings, so we remove them
ticket_matrix_data = ticketmatrix[1:]

K=20 #four clusters
num_customers = len(ticket_matrix_data) #assume listed offers are distinct
num_tickets = 0
for col in ticket_matrix_data:
    num_tickets += 1
#create a num_deals x num_customers matrix of which customer took which deal
ticket_matix = np.zeros((num_customers,num_customers))
ticket_matrix = ticket_matrix_data

#initialize and carry out clustering
km = KMeans(n_clusters = K)
km.fit(ticket_matrix)

#find center of clusters
centers = km.cluster_centers_
centers[centers<0] = 0 #the minimization function may find very small negative numbers, we threshold them to 0
centers = centers.round(2)
print('\n--------Centers of the four different clusters--------')
print('Deal\t Cent1\t Cent2\t Cent3\t Cent4')
for i in range(num_deals):
    print(i+1,'\t',centers[0,i],'\t',centers[1,i],'\t',centers[2,i],'\t',centers[3,i])

#find which cluster each customer is in
prediction = km.predict(ticket_matrix_data)
print('\n--------Which cluster each customer is in--------')
print('{:<15}\t{}'.format('Customer','Cluster'))
for i in range(len(prediction)):
    print('{:<15}\t{}'.format(ticket_matrix_data[i][1],prediction[i]+1))
'''
#determine which deals are most often in each cluster
deal_cluster_matrix = np.zeros((num_deals,K),dtype=np.int)
print('\n-----How many of each deal involve a customer in each cluster-----')
print('Deal\t Clust1\t Clust2\t Clust3\t Clust4')            
for i in range(deal_number):
    for j in range(cust_number):
        if deal_customer_matrix[i,j] == 1:
            deal_cluster_matrix[i,prediction[j]] += 1

for i in range(deal_number):
    print(i+1,'\t',end='')
    for j in range(K):
        print(deal_cluster_matrix[i,j],'\t',end='')
    print()
print()
'''
print('The total distance of the solution found is',sum((km.transform(customer_deal_matrix)).min(axis=1)))