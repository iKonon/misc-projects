There are a lot of papers to analyze users by their web cites visits while more than half
of digital traffic online now comes from mobile devices and through mobile apps (based
on [comScore report](http://www.comscore.com/Insights/Blog/Major-Mobile-Milestones-in-May-Apps-Now-Drive-Half-of-All-Time-Spent-on-Digital)).

The goal is to predict the demographic and life style profiles of users based on their
previous locations and past behavior at a certain hour of a day.
In case if we have additional context (like any truth set, or application used, user’s
tweets, etc.) we could tune the model.

As a first step, let’s imagine we have a data set that contains user id, timestamp and
location (latitude/longitude pair).

1) Detect “frequent spots”:
<ul>
<li>cluster data using KMeans algorithm (represent users trajectories as fixed-length
vectors of coordinates and then compare such vectors by means of Euclidean
distance) or (as another approach) using Hidden Markov models</li>
<li>detect multiple interleaved periods using Fourier Transform and autocorrelation</li>
</ul>

| record    | user    | timestamp           | latitude  | longitude   |
|-----------|---------|---------------------|-----------|-------------|
| $r_1$     | $u_1$   | 2016-05-09 09:00:00 | 37.786137 | -122.409143 |
| $r_2$     | $u_1$   | 2016-05-09 09:30:00 | 37.785737 | -122.410922 |
| $r_3$     | $u_1$   | 2016-05-09 13:00:00 | 37.787011 | -122.406039 |
| $r_4$     | $u_2$   | 2016-03-26 12:45:00 | 37.786200 | -122.40960  |
| $r_5$     | $u_3$   | 2016-03-01 17:15:00 | 37.785934 | -122.411144 |

2) Label the spots based on timestamps and external context available (like type of location from GooglePlacesAPI): “Office”, “Home”, “Shopping Mall” etc.

| record    | annotation                                           |
|-----------|------------------------------------------------------|
| $r_1$     | San Francisco, Starbucks, coffeehouse, working hours |
| $r_2$     | San Francisco, Hilton, hotel, working hours          |
| $r_3$     | San Francisco, Macy's, department store, lunch time  |
| $r_4$     | San Francisco, road                                  |
| $r_5$     | San Francisco, FedEx                                 |

3) Predict user profiles using decision trees with generative grammar component (associative rules, NLP are applicable).

##### High-level examples
<ul>
<li>Frequent visits to “Victoria Secret” => Gender: female</li>
<li>Frequent visits to Chinese, Japanese restaurants => Food interest: Asian</li>
</ul>

Let’s consider a finite set of users $U$, a finite set of profiles $P$ and describe a finite set of rules $u_i \rightarrow \phi$, where $u_i \in U$ and $\phi \in P$.

##### Example
Suppose we have users $(u_1, u_2, u_3, u_4, u_5)$ and the following rules:

| Conditional rules                                    | Decision rules                                            |
|------------------------------------------------------|-----------------------------------------------------------|
| $u_1 \mid (s_1=”+” \wedge s_2=”-”) \rightarrow \phi_{11}$ | $u_1 \mid (\phi_1 = \phi_{11}) (s_3 := “+”)$         |
| $u_1 \mid  (s_1=”+” \wedge s_2=”+”) \rightarrow \phi_{12}$ | $u_1 \mid (\phi_1 = \phi_{12}) (s_3:= “-“)$         |
| $u_2 \mid (s_3=”+” \wedge s4=”+”) \rightarrow \phi_{21}$  | $u_2 \mid \phi_2 = \phi_{21}) (s_5=”+”)$             |
| $u_3 \mid (s_4=”+”) \rightarrow \phi_{31}$              | $u_3 \mid (\phi_3 = \phi_{31}) (s_2:= “-“)$            |
| $u_3 \mid (s_4=”-”) \rightarrow \phi_{32}$              |                                                        |
| $u_4 \mid (s_6=”+”) \rightarrow \phi_{41}$              | $u_4 \mid (\phi_4 = \phi_{41}) (s_1:= “-“)$            |
| $u_4 \mid (s_6=”-”) \rightarrow \phi_{42}$        | $u_4 \mid (\phi_4 = \phi_{42}) (s_1:= “+“ \wedge s_4:= “+“)$ |
| $u_5 \mid (s_1=”+”) \rightarrow \phi_{51}$              | $u_5 \mid (\phi_5 = \phi_{51}) (s_6:= “-“)$            |
| $u_5 \mid (s_1=”-”) \rightarrow \phi_{52}$              | $u_5 \mid (\phi_5 = \phi_{52}) (s_6:= “+“)$            |

Then the algorithm is as follows:

| Setting |     |     |     |     |    | Profile Ai, Rule type | Hypothesis                                     |
|---------|-----|-----|-----|-----|----|-----------------------|------------------------------------------------|
| $s_1$   |$s_2$|$s_3$|$s_4$|$s_5$| s6 | I                     |                                                |
| .       | .   | .   | .   | .   | .  | -                     |                                                |
| +       | -   | .   | .   | .   | .  | 1, cond.              | $H_1$: $s_1=”+” \wedge s_2=”-”$                |
| +       | -   | +   | .   | .   |    | 1, cond.              |                                                |
| +       | -   | +   | +   | .   | .  | 2, cond.              | $H_2$: $s_4=”+”$                               |
| +       | -   | +   | +   | +   | .  | 2, cond.              |                                                |
| +       | -   | +   | +   | +   | .  | 3, cond.              |                                                |
| +       | -   | +   | +   | +   | .  | 3, cond.              | Confirmation for $s_2=”-”$ in $H_1$            |
| +       | -   | +   | +   | +   | +  | 4, cond.              | $H_3$: $s_6=”+”$                               |
| -       | -   | +   | +   | +   | +  | 4, cond.              | Rejection for $s_1=”+”$ in $H_1$               |
| +       | -   | +   | +   | +   | -  | 4, cond.              | $H_3$: $s_6=”-”$                               |
| +       | -   | +   | +   | +   | -  | 4, cond.              | Confirmation for $s_1=”+”$ in $H_1$ and $s_4=”+”$ in $H_2$ |
| +       | -   | +   | +   | +   | -  | 5, cond.              |                                                |
| +       | -   | +   | +   | +   | -  | 5, cond.              | Confirmation for $s_6=”-”$ in $H_3$            |

Therefore we obtain the following classification:

| $u_1$       | $u_2$       | $u_3$       | $u_4$       | $u_5$       |
|-------------|-------------|-------------|-------------|-------------|
| $\phi_{11}$ | $\phi_{21}$ | $\phi_{31}$ | $\phi_{42}$ | $\phi_{51}$ |

Improvements and known issues:
<ul>
<li>GPS accuracy. The United States government currently [claims](http://www.gps.gov/systems/gps/performance/accuracy/) 4 meter RMS (7.8
meter 95% Confidence Interval) horizontal accuracy for civilian (SPS) GPS.
Vertical accuracy is worse. So in step 2, we need to use not latitude/longitude
pair, but a circle with radius at least 8 meters (we choose 10 meters).</li>
<li>For demographic profiles some open data sets can be used as the truth sets like:
<ul>
<li>http://proximityone.com/location_based_demographics.htm</li>
<li>http://www.census.gov/topics/income-poverty/income.html</li>
</ul>
</li>
<li>To smooth our probabilities in case of high deviations it worst to add some
weights to every profile. As a first approach, for this step we need to estimate the
overall population in the area using [deep learning model](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004845).</li>
</ul>
