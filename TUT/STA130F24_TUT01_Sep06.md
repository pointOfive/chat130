# STA130 TUT 01 (Sep06)<br><br> 🏃🏻‍♀️ 🏃🏻 <u> Hitting the ground running... <u>


### 🚧 🏗️ (Using notebooks and ChatBots) Demo [45 minutes]  
      
#### 1. *[About 8 of the 45 minutes]* Demonstrate going to the course [Quercus homepage](https://q.utoronto.ca/courses/354091); accessing the [Course GitHub Repo](https://github.com/pointOfive/STA130_ChatGPT); opening new and uploaded notebooks on [UofT Jupyterhub](https://datatools.utoronto.ca) (classic jupyter notebook, or jupyterhub is fine, or students may use [google collab](https://colab.research.google.com/)); and using Jupyter notebooks as a "`Python` calculator" and editing ["Markdown cells"](https://www.markdownguide.org/cheat-sheet/)<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> This is all simple and pretty obvious intended to be that way so students get that they can do this on their own.

</details>
    
#### 2. *[About 30 of the 45 minutes]* Demonstrate using [ChatGPT](https://chat.openai.com/) (or [Copilot](https://copilot.microsoft.com/) if conversation privacy is desired) to

1. find an (a) "amusing, funny, or otherwise interesting dataset" which (b) has missing values and is (c) available online through a URL link to a csv file;
2. load the data into the notebook with `pandas` and get "missing data counts" for the dataset;
3. prompt the ChatBot to "Please provide a summary of our interaction for submitting as part of the requirements of an assignment"; and, to "Please provide me with the final working verson of the code that we created"<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>

> The intention here is to demonstrate that [at the current GPT4.0-ish level] the ChatBot (with probably about an 80% chance) **cannot fullfil all the requests of the inquiry (of (a) "funny or amusing" nature, (b) the presence of missingness, and (c) working url links)** *but will otherwise produce working code*<br><br>
> 
> 1. ChatBots have a notoriously "short term memory"; so, be ready for them to "forget" specific details of your prompting requests 
> 2. ChatBots often cannot pivot away substantially from initial answers; so, be ready for your efforts at follow up and correction with the ChatBot to prove frustratingly futile (which, may in this case actually have a lot to do with the following fact, that...)
> 3. ChatBots don't seem to be very aware of the contents of datasets that are avalable online (or even working url links where datasets are); so, ChatBot are not currently a substitue for exploring dataset repository such as [TidyTuesday](https://github.com/rfordatascience/tidytuesday) (or other data repositiory resources) and reviewing data yourself (although, ChatBot interactions can nonetheless be help with brainstorm dataset ideas and provide a way to "search for content", perhaps especially when referencing a specific website in the conversation)<br><br>
> 
> Examples of this task going pretty well are available [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00006_copilot_funnyamusingNAdatasetV3.md), [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00007_copilot_funnyamusingNAdatasetV4.md), and [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00001_gpt3p5_villagersdata.md); while, examples of this going poorly are available [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00002_copilot_funnyamusingNAdataset.md) and [here](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00002_gpt3p5_funnyasusingNAdataset.md). Successes and failures are found within the Microsoft Copilot and ChatGPT ChatBots both, suggesting the quality of the results likely has to do more with "randomness" and perhaps the nature of the prompting and engagement as opposed to the actual ChatBot version being used...
    
</details>

#### 3. *[About 7 of the 45 minutes]* Demonstrate saving your python jupyter notebook in your own account and "repo" on [github.com](https://github.com), and sharing (a) notebook links, (b), ChatBot transcript log links, (c) ChatBot summaries through a piazza post and a Quercus announcement (so students can use this later for their homework assignment if they wish)<br><br>


### 💬 🗣️ Communication [55 minutes]  
     
#### 1. *[About 15 of the 55 minutes]* Ice breakers  and introductions, in 8 groups of 3 or thereabouts...
    
1. Each person may bring two emojis to a desert island... reveal your emojis at the same time... for emojis selected more than once the group should select one additional emoji
2. Where are you from, what do you think your major might be, and what's an "interesting" fact that you're willing to share about yourself?
        
#### 2. *[About 10 of the 55 minutes]* These are where all the bullet holes were in the planes that returned home so far after some missions in World War II
    
1. Where would you like to add armour to planes for future missions?
2. Hint: there is a hypothetical dataset of the bullet holes on the planes that didn't return which is what we'd ideally compare against the dataset we observe...
        
![Classic image of survivorship bias of WW2 planes](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Survivorship-bias.svg/640px-Survivorship-bias.svg.png)
           
#### 3. *[About 10 of the 55 minutes]* Monte Hall problem: there is a gameshow with three doors, one of which has a prize, and you select one of the doors and the gameshow host reveals one of the other two unchosen doors which does not have the prize... would you like to change your guess to the other unchosen door?

![](https://mathematicalmysteries.org/wp-content/uploads/2021/12/04615-0sxvwbnzvvnhuklug.png)<br>
       
#### 4. *[About 20 of the 60 minutes]* Discuss the experience of the groups for the WW2 planes and Monte Hall problems

1. For each of these problems, have students vote on whether their groups (a) agreed on answers from the beginning, (b) agreed only after some discussion and convincing, or (c) retained somewhat divided in their opinions of the best way to proceed<br><br>
    
2. Briefely identify the correct answer from the answers the groups arrived at<br><br>
    
3. **[If time permits... otherwise this is something students could consider after TUT]** Prompt a [ChatGPT](https://chat.openai.com/) [or [Copilot](https://copilot.microsoft.com/)] ChatBot to introduce and explain "survivorship bias" using spotify songs as an example and see if students are able to generalize this idea for the WW2 planes problem and if they find it to be a convincing argument to understand the problem<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> This could be done like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00009_copilot_survivorshipbias_spotify.md) or [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00003_gpt3p5_spotify_Survivorship_Bias.md), or you could instead try to approach things more generally like [this](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00004_gpt3p5_general_Survivorship_Bias.md)
> 
> Two ends of the ChatBot prompting spectrum are
> 
> 1. creating an extensive prompt exhuastively specifying the desired response results; or, 
> 2. iteratively clarifying the desired response results through interactive ChatBot dialogue<br><br>
> 
> This is to some degree a matter of preference regarding the nature of ChatBot conversation sessions, but there it may also be a lever to influence the nature of the responses provided by the ChatBot 
</details>
    
4. **[If time permits... otherwise this is something students could consider after TUT]** Prompt a [ChatGPT](https://chat.openai.com/) [or [Copilot](https://copilot.microsoft.com/)] ChatBot to introduce and explain the Monte Hall problem and see if the students find it understandable and convincing<br>

<details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
    
> ChatBots fail to correctly analyze the Monte Hall problem when they're asked for a formal probabilistic argument...
>
> - [ChatGPT fails by wrongly calculating a probability of 1/2...](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/GPT/SLS/00005_gpt3p5_MonteHallWrong.md)
> - [Copilot fares similarly poorly without substantial guidance...](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk1/COP/SLS/00010_copilot_montehallwrong.md)<br><br>
> 
> *demonstrating (a) that there are clear limits to how deeply ChatBots actually "reason", and (b) that they are instead better understood as simply being information regurgitation machines, and (c) that this means  they can suffer from the "garbage in, garbage out" problem if the quality of the information their responses are based on are is poor and inaccurate (as is notoriously the case in the Monte Hall problem, for which many incorrect mathematical analyses have been "published" into the collection of human generated textual data on which ChatBots are based)*
    
</details>


```python

```
