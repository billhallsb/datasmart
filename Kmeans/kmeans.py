

# -*- coding: utf-8 -*-
"""
A program to carry out Kmeans clustering where K=4
on data relating to wine marketing from book 
"Data Smart: Using Data Science to Transform Information into Insight"

Requires csv input file OfferInfo.csv with headings
'Campaign', 'Varietal', 'Minimum Qty (kg)', 'Discount (%)', 'Origin', 'Past Peak'
and input file Transactions.csv with headings
'Customer Last Name', 'Offer #'
"""

#make more similar to Python 3
from __future__ import print_function, division, absolute_import#, unicode_literals

#other stuff we need to import
import csv
import sys
import numpy as np
from sklearn.cluster import KMeans
reload(sys)
sys.setdefaultencoding("utf-8")

#beginning of main program

#read in OfferInfo.csv
csvf = open('rawticket.csv','rU')
rows = csv.reader(csvf)
rawticket_sheet = [row for row in rows]
csvf.close()
'''
#read in Transactions.csv
csvf = open('Transactions.csv','rU')
rows = csv.reader(csvf)
transaction_sheet = [row for row in rows]
csvf.close()
'''
#first row of each spreadsheet is column headings, so we remove them
rawticket_data = rawticket_sheet[1:]
#transaction_sheet_data = transaction_sheet[1:]

K=20 #twenty clusters
#num_deals = len(offer_sheet_data) #assume listed offers are distinct
ticket_types =[]
for row in rawticket_data:
    ticket_types.append(row[10])
ticket_types = list(set(ticket_types))
ticket_types.sort()
num_tickettypes = len(ticket_types)
#find the sorted list of customer last names
customer_names = []
for row in rawticket_data:
    customer_names.append(row[21])
customer_names = list(set(customer_names))
customer_names.sort()
num_customers = len(customer_names)

#create a num_deals x num_customers matrix of which customer took which deal
deal_customer_matrix = np.zeros((num_tickettypes,num_customers))
for row in rawticket_data:
    cust_number = customer_names.index(row[21])
    deal_number = ticket_types.index(row[10])
    deal_customer_matrix[deal_number-1,cust_number] += 1
customer_deal_matrix = deal_customer_matrix.transpose()

#initialize and carry out clustering
km = KMeans(n_clusters = K)
km.fit(customer_deal_matrix)

#find center of clusters
centers = km.cluster_centers_
centers[centers<0] = 0 #the minimization function may find very small negative numbers, we threshold them to 0
centers = centers.round(2)
'''
print('\n--------Centers of the four different clusters--------')
print('Deal\t Cent1\t Cent2\t Cent3\t Cent4\tCent5')
for i in range(num_tickettypes):
    print(i+1,'\t',centers[0,i],'\t',centers[1,i],'\t',centers[2,i],'\t',centers[3,i])
'''
#find which cluster each customer is in
prediction = km.predict(customer_deal_matrix)
print('\n--------Which cluster each customer is in--------')
print('{:<15}\t{}'.format('Customer','Cluster'))
for i in range(len(prediction)):
    print('{:<15}\t{}'.format(customer_names[i],prediction[i]+1))

with open('cluster.csv', 'wb') as csvfile:
    predictwriter = csv.writer(csvfile, delimiter=' ')
    for i in range(len(prediction)):
        predictwriter.writerow('{:<15}\t{}'.format(customer_names[i],prediction[i]+1))
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
'''
print()

print('The total distance of the solution found is',sum((km.transform(customer_deal_matrix)).min(axis=1)))