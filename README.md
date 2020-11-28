# Understanding Capital Bikeshare Ridership

Created by Aren Carpenter

Fall 2020

## Introduction

After some more advanced neural network projects focusing mainly on computer vision, I wanted to return to the data scientist's bread and butter, generating business insights from data through EDA and rigorous hypothesis testing. Utilizing the [Capital Bikeshare rider data](https://data.world/data-society/capital-bikeshare-2011-2012) from 2011-2012, I will illuminate the differences between registered and casual bikers and how various environmental and temporal features affect these populations.

With a climate crisis looming and alternative methods of personal transportation gaining popularity, short-term bike rentals through companies like CitiBike and Capital Bike offer users easy access to bikes. The main cohort of registered renters are business professionals on their morning and evening commutes, while a casual cohort utilizes the bikes for leisure mostly on weekends.

## Repo Navigation

- **[01_Exploratory_Data_Analysis](01_Exploratory_Data_Analysis.ipynb)**: Contains EDA and data visualizations
- **[02_Hypothesis_Testing](02_Hypothesis_Testing.ipynb)**: Contains hypothesis testing to illuminate 
- **[Images](Images/)**: Directory for images/visualizations

## Exploratory Data Analysis

The data was collected and some small cleaning was required, mainly for time series analysis. Some types were recast, but, unlike most real-world data, this set was mostly intact. I wanted to focus on creating interesting and useful visualizations, but in the future will feature an incredibly messy data cleaning project.

### Data Visualizations

Let's start by looking at some aggregated ridership patterns. 

#### Total users throughout the year

We see rider data from both 2011 and 2012. As we expect, there are more riders in the late spring through early fall when the temperatures and weather are the most comfortable. Ridership begins to fall off in November through April. Overall ridership is higher in 2012 than in 2011 (INSERT STATS HERE), perhaps signaling greater adoption and market share of Capital bike in Washington, D.C.

![](Images/Users_by_Year.png) 

#### Total users by month

Just reinforcing the previous time series evaluation resampled by month, showing higher ridership in the spring and summer. However, grouping by rider population begins to show the differences between registered and casual cohorts that we will continue to explore below. Registered users account for a majority of the rides regardless of month.

![](Images/Users_by_Month.png)

#### Total users by day of the week

Here's our starkest difference between our two cohorts. Registered users ride mostly duing the work week for their commutes, while casual users ride mostly on the weekends. If you don't separate the two populations than ridership is almost perfectly equal across all days, thus showcasing how understanding customer segmentation in your data is very important.

![](Images/Users_by_Day.png)

#### Rides by hour of the day

Looking at ridership by hour in the day also illustrates the main cohort of registered users renting for their commutes as we see peaks around 8 am and 5-6 pm. 

![](Images/Ridership_by_Hour.png)

#### "Feels like" temperature distribution

We'll use temperature for some feature engineering later, but looking at the "feels like" temperature we see a wide range of temperatures at which users have rented bikes, from about 0 degrees to 120 degrees. This accounts for humidity, windage, etc. and is more extreme than raw temperatures, but provides a better sense of what the rider is actually experiencing. 

![](Images/Feels_Like_Temp_Dist.png)

#### Ridership by weather type

Regardless of the weather, registered users ride more than casual users. Within groups, there are more rides in good weather that declines as weather worsens. Interestingly, registered users still rent at relatively high rates during the worse weather (classified as "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"). 

![](Images/Users_by_Weather_Type.png)

#### Ridership on holidays

There are more rides by registered users not during holidays, likely just because more days are not holidays and the average American worker only gets about 10 holiday days off. Casual users, by contrast, are more likely to ride on holidays than normal days. 

![](Images/Users_by_Holiday.png)

### Hypothesis Testing

#### Percentage of rides during rush hour?

## Conclusion and Next Steps