import streamlit as st
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import os
import time
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(page_title="MaxtekAI",layout="wide")
st.title("Marketing analytics web App running on behavioral clustering algorithm")
st.subheader('AI Web App by [Maximilien Kpizingui](https://kpizmax.hashnode.dev)')
'''
Hey there! Welcome to Maximilien's marketing analytics  App. This app
analyzes (and never stores!)
your company customer data throughout the year to optimize your marketing strategy.
Give it a go by
uploading your data!
'''
class SomeModel():
	def __init__(self):
		pass
	def get_preprocessing(_self, a, b , c,d):
 	#	removing duplicated index and dropping nan values
		X= pd.read_csv(d).drop_duplicates(keep="first")
		X=X[pd.notnull(X[a])]
		X=X[pd.notnull(X[b])]
		X=X[pd.notnull(X[c])]
		return X
	
	def plotCountry(_self,a,b,c,d):
		preprocessing_df = _self.get_preprocessing(a,b,c,d)
		new_df= preprocessing_df.groupby(by=["location_country"]).count()[["total_pageviews"]]
		st.write("Total page views per country")
		fig = plt.figure(figsize=(10, 8))
		fig = px.bar(new_df)
		st.plotly_chart(fig)
		st.write("")
	
	def get_rfm_modeling(self, a,b,c,d):
        # function to return the rfm dataframe
		preprocessed_df = self.get_preprocessing( a, b, c,d)
		df_recency = preprocessed_df.groupby(by='location_country',as_index=False)['total_sessions'].sum()
		df_recency.columns = ['location_country', 'Recency'] 
		frequency_df = preprocessed_df.drop_duplicates().groupby( by=['location_country'], as_index=False)['total_conversion'].count()
		frequency_df.columns = ['location_country', 'Frequency']
		preprocessed_df['Total'] =preprocessed_df['total_conversion']*preprocessed_df['total_carts']
		monetary_df = preprocessed_df.groupby(by='location_country', as_index=False)['Total'].sum()
		monetary_df.columns = ['location_country', 'Monetary']
		rf_df = df_recency.merge(frequency_df, on='location_country')
		rfm_df = rf_df.merge(monetary_df, on='location_country')
		return rfm_df
	
	def R_score(self,var,p,d):
        # recency score on 2h activity high value, more logs on the platform
		if var <= d[p][0.25]:
			return 1
		elif var <= d[p][0.50]:
			return 2
		elif var <= d[p][0.75]:
			return 3
		else:
			return 4
	
	def FM_score(self,var,p,d):
#Frequency and Monetary score (Positive Impact : Higher the value, better the customer)   
		if var <= d[p][0.25]:
			return 4
		elif var <= d[p][0.50]:
			return 3
		elif var <= d[p][0.75]:
			return 2
		else:
			return 1
	
	def get_rfmscore(self,a,b,c,d):
#Segmentation: Here, we will divide the data set into 4 parts based on the quantiles.
		rfm_df = self.get_rfm_modeling(a,b,c,d)
		quantiles =rfm_df.drop('location_country',axis = 1).quantile(q = [0.25,0.5,0.75])
		rfm_df['R_score'] = rfm_df['Recency'].apply(self.R_score,args = ('Recency',quantiles,))
		rfm_df['F_score'] = rfm_df['Frequency'].apply(self.FM_score,args = ('Frequency',quantiles,))
		rfm_df['M_score'] = rfm_df['Monetary'].apply(self.FM_score,args = ('Monetary',quantiles,))
        #Now we will create : RFMGroup and RFMScore
		rfm_df['RFM_Group'] = rfm_df['R_score'].astype(str) + rfm_df['F_score'].astype(str) + rfm_df['M_score'].astype(str)
        #Score
		rfm_df['RFM_Score'] = rfm_df[['R_score','F_score','M_score']].sum(axis = 1)
		rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
		rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
		rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)
        # normalizing the rank of the customers
		rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
		rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
		rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100
		rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
		rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm']+0.28 * rfm_df['F_rank_norm']+0.57*rfm_df['M_rank_norm']
		rfm_df['RFM_Score'] *= 0.05
		rfm_df = rfm_df.round(2)
		return rfm_df

	
	def get_customerSegment(self,a,b,c,d):
		rfm_df = self.get_rfmscore(a,b,c,d)
		rfm_df["Customer_segment"] = np.where(rfm_df['RFM_Score'] > 4.5, "Top Customers",
                             (np.where(rfm_df['RFM_Score'] > 4,"High value Customer",
                             (np.where(rfm_df['RFM_Score'] >= 3,"Medium Value Customer",
                             np.where(rfm_df['RFM_Score'] > 1.6,'Low Value Customers', 'Low Customers'))))))
		return rfm_df
	
	def right_treat(self,var):
        # First will focus on the negative and zero from the dataset before the transformation.
		if var <= 0:
			return 1
		else:
			return var
	
	def get_screwLogTransform(self,a,b,c,d):
		rfm_df = self.get_customerSegment(a,b,c,d)
#skewness transform
		rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x : self.right_treat(x))
		rfm_df['Monetary'] = rfm_df['Monetary'].apply(lambda x : self.right_treat(x))
#Log Transformation
		log_RFM_data = rfm_df[['Recency','Frequency','Monetary']].apply(np.log,axis = 1).round(4)
		return log_RFM_data
	
	def scaledLogTransform(self,a,b,c,d):
		rfm_data = self.get_screwLogTransform(a,b,c,d)
		MinScaler = MinMaxScaler()
		scaled_RFM_data =MinScaler.fit_transform(rfm_data)
		scaled_RFM_data = pd.DataFrame(scaled_RFM_data,columns=rfm_data.columns,index=rfm_data.index)
		return scaled_RFM_data
	
	def plotClusteringElbow(self,a,b,c,d):
		x = self.scaledLogTransform(a,b,c,d)
		model = KMeans()
		visualizer =KElbowVisualizer(model, k=(1,9))
		visualizer.fit(x)
		st.write(visualizer.show())
	@st.cache
	def train(self,a,b,c,d):
		scaled_data = self.scaledLogTransform(a,b,c,d)
		KM_clust = KMeans(n_clusters= 3, init = 'k-means++',max_iter = 1000,random_state=42)
		KM_clust.fit(scaled_data)
		return KM_clust
	
	def get_df(self):
		shopify_file = st.file_uploader('Select Your Local shopify  test_csv (default provided)')
		if shopify_file is not None:
			shopify_df = pd.read_csv(shopify_file)	
			st.write('csv file loaded successfully')
			X= shopify_df.drop_duplicates(keep="first")
			X=X[pd.notnull(X['location_country'])]
			X=X[pd.notnull(X['referrer_source'])]
			X=X[pd.notnull(X['referrer_name'])]
			df_recency = X.groupby(by='location_country',as_index=False)['total_sessions'].sum()
			df_recency.columns = ['location_country', 'Recency']
			frequency_df = X.drop_duplicates().groupby( by=['location_country'], as_index=False)['total_conversion'].count()
			frequency_df.columns = ['location_country', 'Frequency']
			X['Total'] =X['total_conversion']*X['total_carts']
			monetary_df = X.groupby(by='location_country', as_index=False)['Total'].sum()
			monetary_df.columns = ['location_country', 'Monetary']
			rfm_df = df_recency.merge(frequency_df, on='location_country')
			rfm_df = rfm_df.merge(monetary_df, on='location_country')
			rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x : self.right_treat(x))
			rfm_df['Monetary'] = rfm_df['Monetary'].apply(lambda x : self.right_treat(x))
#Log Transformation
			log_RFM_data = rfm_df[['Recency','Frequency','Monetary']].apply(np.log,axis = 1).round(4)
			MinScaler = MinMaxScaler()
			scaled_RFM_data =MinScaler.fit_transform(log_RFM_data)
			scaled_RFM_data = pd.DataFrame(scaled_RFM_data,columns=log_RFM_data.columns,index=log_RFM_data.index)
			return scaled_RFM_data,rfm_df,X

		else:
			st.warning("Warning. Waiting for csv file")
			st.balloons()
			st.progress(80)
			with st.spinner('Wait for it...'):
				time.sleep(10)
			st.stop()
 #		model=self.train(a,b,c)
#		new_scaled_data = self.scaledLogTransform(a,b,c)
#		y_predict= model.fit_predict(new_scaled_data)
#		pass
	
	def get_results(self,a,b,c,d):
		model=self.train(a,b,c,d)
		rfm_df = self.get_customerSegment(a,b,c,d)
		rfm_df['Cluster'] = model.labels_
		rfm_df['Cluster'] = 'Cluster' + rfm_df['Cluster'].astype(str)
		new_rfm_df =  rfm_df[["location_country","RFM_Score","Customer_segment"]]
		return new_rfm_df
	@st.cache
	def get_results_cluster(self,a,b,c,d):
                model=self.train(a,b,c,d)
                rfm_df = self.get_customerSegment(a,b,c,d)
                rfm_df['Cluster'] = model.labels_
                rfm_df['Cluster'] = 'Cluster' + rfm_df['Cluster'].astype(str)
                new_rfm_df =  rfm_df[["location_country","Customer_segment","Cluster"]]
                return new_rfm_df
        
	@st.cache
	def get_results_plot(self,a,b,c,d):
                model=self.train(a,b,c,d)
                rfm_df = self.get_customerSegment(a,b,c,d)
                rfm_df['Cluster'] = model.labels_
                rfm_df['Cluster'] = 'Cluster' + rfm_df['Cluster'].astype(str)
                return rfm_df
	
	def plotgraph(self,a,b,c,d):
		col1, col2 = st.columns(2)
		plot_data =self.get_results_plot(a,b,c,d)
		data=self.get_results(a,b,c,d)
		data_cluster =self.get_results_cluster(a,b,c,d)
		with col1:
			new_data= data[["location_country","Customer_segment"]]
			new_df= new_data.groupby(by=["Customer_segment"]).count()[["location_country"]]
			mylabels = ["Cluster0", "Cluster1", "Cluster2"]
			fig = plt.figure(figsize=(10, 4))
			st.write("Customer segmentation bar chart")
#			st.bar_chart(new_df)
			fig = px.bar(new_df)
			st.plotly_chart(fig)
			st.write("")
		with col2:
			st.write("Customer cluster bar chart")
			fig = px.histogram(data_cluster["Cluster"])
			st.plotly_chart(fig)
			st.write("")
		col3, col4 =st.columns(2)
		with col3:
			st.write("Customer segmentation scatter plot based on rfm score")
			fig2, ax = plt.subplots()
			data_plot2=self.get_results_plot(a,b,c,d)
			fig = px.scatter(data_plot2,x = "Monetary", y = "Frequency",color = "Customer_segment")
			st.plotly_chart(fig)
		with col4:
			st.write("Customer clustering scatter plot based on rfm score")
			fig = px.scatter(data_plot2,x = "Monetary", y = "Frequency",color = "Cluster")
			st.plotly_chart(fig)

if __name__ == '__main__':
	global shopify_file
	st.image("banner.jpg")
	page = st.sidebar.selectbox(
	"Select a Page",
	[
		"Upload csv file",
		"Home page",
		"RFM model",
		"Customer segmentation",
		"Customer clustering",
		"Data visualization",
		"Prediction"
	])

	st.sidebar.markdown("Email: maximilien@tutanota.de")
	st.sidebar.markdown("Element ID: @maximilien:matrix.org")
	if page == "Upload csv file":
		st.subheader('Please download the dataset on my blog using this [Link](https://kpizmax.hashnode.dev)')
		shopify_file = st.file_uploader('Select Your Local shopify  train_csv (default provided)')
		if shopify_file is not None:
        		shopify_df = pd.read_csv(shopify_file)
        		st.write('csv file loaded successfully')
		else:
			st.balloons()
			st.progress(80)
			with st.spinner('Wait for it...'):
    				time.sleep(10)
			st.warning("Warning. Waiting for csv file")
			st.stop()

	model_instance = SomeModel()
	if page == "Home page":
		st.balloons()
		st.subheader("Home page")
		st.write(model_instance.get_preprocessing('location_country','referrer_source','referrer_name','shopify_data_seller1.csv'))
		st.subheader('Pandas Profiling of shopify  Dataset')
		shopify_profile = ProfileReport(model_instance.get_preprocessing('location_country','referrer_source','referrer_name','shopify_data_seller1.csv'), explorative=True)
		st_profile_report(shopify_profile)
		st.write("")
#		st.write(model_instance.plotCountry('location_country','referrer_source','referrer_name','shopify_data_seller1.csv'))
	if page == "RFM model":
		st.subheader("RFM model")
		st.balloons()
		st.write(model_instance.get_rfm_modeling("location_country","referrer_source","referrer_name",'shopify_data_seller1.csv'))
#	model_instance.get_customerSegment("location_country", "referrer_source", "referrer_name",'shopify_data_seller1.csv')
#	model_instance.get_screwLogTransform("location_country","referrer_source","referrer_name",'shopify_data_seller1.csv')
#	model_instance.scaledLogTransform("location_country","referrer_source", "referrer_name",'shopify_data_seller1.csv')
#	model_instance.train("location_country","referrer_source", "referrer_name",'shopify_data_seller1.csv')
	st.write("")
	if page == "Customer segmentation":
		st.header("Customer segmentation")
		st.write(model_instance.get_results("location_country", "referrer_source","referrer_name",'shopify_data_seller1.csv'))
	if page == "Customer clustering":
                st.header("Customer clustering")
                st.write(model_instance.get_results_cluster("location_country", "referrer_source","referrer_name",'shopify_data_seller1.csv'))
	if page == "Data visualization":
		st.balloons()
		st.header("Data visualization")
		st.write(model_instance.plotCountry('location_country','referrer_source','referrer_name','shopify_data_seller1.csv'))
		model_instance.plotgraph("location_country","referrer_source", "referrer_name",'shopify_data_seller1.csv')
	if page == "Prediction":
		new_data,rfm,X = model_instance.get_df()
		new_df = X.groupby(by=["location_country"]).count()[["total_orders_placed"]]
		st.write("Total page views per country")
		fig = plt.figure(figsize=(10, 8))
		fig = px.bar(new_df)
		st.plotly_chart(fig)
		st.write("")
		model = model_instance.train("location_country","referrer_source", "referrer_name",'shopify_data_seller1.csv')
		label = {0:"Cluster 1 -> low value customer",1:"Cluster 0 -> medium value customer ", 2:"Cluster 2 -> high value customer"}
		y_predict= model.fit_predict(new_data)
		predict_df = pd.DataFrame(data=y_predict,columns=['cluster'],index=rfm.index)
		predict_df['cluster']= predict_df['cluster'].map(label)
		location_df = pd.DataFrame(data=rfm,columns=['location_country'])
		final = pd.concat([location_df,predict_df],axis=1,join="inner")
		st.write(final)
		predict_df['cluster']= predict_df['cluster'].map(label)

#	if st.button("Clear all"):
#		st.empty()
