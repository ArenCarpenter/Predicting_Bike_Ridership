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

[](Images/Users by Year.png)

#### Total users by month

Just reinforcing the previous time series evaluation resampled by month, showing higher ridership in the spring and summer. However, grouping by rider population begins to show the differences between registered and casual cohorts that we will continue to explore below. Registered users account for a majority of the rides regardless of month.

[](Images/Users by Month.png)

#### Total users by day of the week

[](Images/Users by Day.png)

#### Rides by hour of the day

[](Images/Ridership by Hour.png)

#### "Feels like" temperature distribution

[](Images/Feels Like Temp Dist.png)

#### Ridership by weather type

[](Images/Users by Weather Type.png)

#### Ridership on holidays

[](Images/Users by Holiday.png)

### Hypothesis Testing

## Conclusion and Next Steps