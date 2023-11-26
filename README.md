# Video_Games_Sales_and_Prediction

About the Project: This project analyzes a dataset of video game sales to provide insights into various aspects of the gaming industry, including regional sales distribution, top-selling genres, and genre sales over time. It utilizes Apache Spark for data processing and Scala as theprogramming language. Apart from that we have built a prediction model for the total global sales using parameters like Name, Platform,
Publisher etc. 

## About the Dataset 
The dataset used for this project is a CSV file containing information about video game sales, including attributes such as Rank(S.No.), Name,
Platform, Year, Genre, Publisher, and sales data for different regions (NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales). The dataset allows
for comprehensive analysis of video game sales trends and patterns. 

The dataset includes games from the year 1980-2020 and was scraped from vgchartz.com. 
Description of the Fields: Rank - Ranking of Games 
Name - The games name 
Platform - Platform of the games release (i.e. PC,PS4, etc.) 
Year - Year of the game’s release 
Genre - Genre of the game 
Publisher - Publisher of the game 
NA_Sales - Sales in North America (in millions) 
EU_Sales - Sales in Europe (in millions) 
JP_Sales - Sales in Japan (in millions) 
Other_Sales - Sales in the rest of the world (in millions) 
Global_Sales - Total worldwide sales. 

## Analysis 
Top Selling Games Regional Breakdown of Top Selling Games Top Selling Games by their Publisher Platform Distribution Games Published each
Year Yearly Sales Distribution Genre Distribution by Sales Genre Distribution Publisher Distribution Regional Market Share for Each Genre Genre
Sales over time Sales by Top Publishers over time Regional Sales over time Rank of each game based on Genre Highest Selling Game of each
genre Top Selling game in each region Average Sales by Genre.

## Prediction Model 
Uses the parameters like Name, Platform, Year, Genre, Publisher to predict the global sales using three different ML algorithms(Random Forest,
Linear Regression, GBT Regression). 
Dependencies 
The project requires the following dependencies: 
Apache Spark: The distributed computing framework used for data processing.
Scala: The programming language used for writing the Spark application.

## Conclusion 
This project provides a comprehensive analysis of video game sales data. It allows for exploration of regional sales distribution, identification of
top-selling genres, and examination of genre sales over time. The project’s insights can be valuable for game developers, publishers, and industry
analysts to make informed decisions and understand market trends.
