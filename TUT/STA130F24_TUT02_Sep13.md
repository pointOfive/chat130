# STA130 TUT 02 (Sep13)<br><br> 👨‍💻 👩🏻‍💻 <u>Coding with data types, for loops, and logical control<u>
    


### ♻️ 📚 Review / Questions [10 minutes]

1. Follow up questions and clarifications regarding regarding **notebooks, markdown, ChatBots, or `Python` code** previously introduced in the Sep06 TUT and Sep09 LEC 

> 1. We're continuing to dive into using **notebooks, markdown**, and `Python` as a tool: building comfort and capability working with **ChatBots** to leverage `Python` code is the objective of the current phase of the course...<br><br>
>
> 2. *Questions about [HW01](https://github.com/pointOfive/stat130chat130/blob/main/HW/STA130F24_HW01_DueSep12.ipynb) which was due yesterday (Thursday Sep12) should be asked in OH from 4-6PM ET Tuesday, Wednesday, and Thursday (see the [Quercus course homepage](https://q.utoronto.ca/courses/354091) for further details)*<br><br>
> 
> 3. *And same story, and even moreso for questions about [HW02](https://github.com/pointOfive/stat130chat130/blob/main/HW/STA130F24_HW02_DueSep19.ipynb) which was due next Thursday (Sep19) (e.g., regarding the "pre-lecture" questions...)*

### 🚧 🏗️ Demo (using Jupyter Notebook and ChatBots) [50 minutes]


#### 1. **[35 of the 50 minutes]** Demonstrate (using a ChatBot to show and explain?) some traditional `python` coding structures

1. `tuple()`, `list()`, vs `dict()  # immutable and mutable "lists" vs key-value pairs`<br><br>
    
2. some `NumPy` functions:<br><br>
    
    1. `import numpy as np`
    2. `np.array([1,2,3]) # a faster "list"`
    3. `np.random.choice([1,2,3])`<br><br>
        
3. `for i in range(n):`, `for x in a_list:`, `for i,x in enumerate(a_list):`, and `print()`<br><br>
    
    1. `variable` as the last line of a "code cell" in a notebook "prints" the value
    2. but `print()` is needed inside `for` loop if you want to output something<br><br>
        
4. `if`/`else` conditional statements<br><br> 

    1. perhaps with `x in b_list`<br>or `i % 2 == 0` to treat evens/odds differently  (sort of like the infamous "FizzBuzz" problem that some people [can't complete](https://www.quora.com/On-average-what-is-the-proportion-of-applicants-that-cannot-pass-a-simple-FizzBuzz-test-based-on-your-personal-experience-or-on-facts) as part of a coding interview challenge)
    2. note the "similarity" to the `try-except` block structure when that's encountered in the code below since it's used there (despite being a "more advanced" 
        
#### 2. **[5 of the 50 minutes]** Reintroduce the [Monty Hall problem](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk2/GPT/SLS/00001_gpt3p5_MonteHall_ProblemExplanation_v1.md) and see which of the coding structures above you recognize (or do not see) in the Monty Hall simulation code below

#### 3. **[10 of the 50 minutes]** Use any remaining time to start a demonstration of using a ChatBot to (a) understand what the code below is doing and (b) suggest an improved streamlined version of the `for` loop simulation code that might be easier to explain and understand
    
> ChatGPT version 3.5 [was very effective](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk2/GPT/SLS/00003_gpt3p5_MonteHall_CodeDiscussion_v1.md) for (b), while Copilot was shockingly bad on a [first try](../CHATLOG/wk2/COP/SLS/00001_creative_MonteHall_CodeDiscussion_v1.md) but was able to do better [with some helpful guidance](https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk2/COP/SLS/00002_concise_MonteHall_CodeDiscussion_v2.md).


```python
# Cell to demo what the above code does

```


```python
# Add more code cells to keep a record of the demos

```


```python
# Monte Hall Simulation Code -- not the only way to code this, but it's what Prof. Schwartz came up with...

import numpy as np
all_door_options = (1,2,3)  # tuple
my_door_choice = 1  # 1,2,3
i_won = 0
reps = 100000
for i in range(reps):
    secret_winning_door = np.random.choice(all_door_options)
    all_door_options_list = list(all_door_options)
    # take the secret_winning_door, so we don't show it as a "goat" losing door
    all_door_options_list.remove(secret_winning_door)
    try:
        # if my_door_choice was secret_winning_door then it's already removed
        all_door_options_list.remove(my_door_choice)
    except:
        pass
    # show a "goat" losing door and remove it
    goat_door_reveal = np.random.choice(all_door_options_list)
    all_door_options_list.remove(goat_door_reveal)

    # put the secret_winning_door back in if it wasn't our choice
    # we previously removed it, so it would be shown as a  "goat" losing door
    if secret_winning_door != my_door_choice:
        all_door_options_list.append(secret_winning_door)
    # if secret_winning_door was our choice then all that's left in the list is a "goat" losing door
    # if secret_winning_door wasn't our choice then it's all that will be left in the list

    # swap strategy
    my_door_choice = all_door_options_list[0]

    if my_door_choice == secret_winning_door:
        i_won += 1

i_won/reps
```




    0.66777



### 💬 🗣️ Communication [40 minutes]
    
#### 1. **[5 of the 40 minutes]** Quickly execute the "rule of 5" and take 5 minutes to break into 5 new groups of about 5 students and assign the following 5 questions to the 5 groups (and note that this instruction uses fives 5's 😉). Consider allowing students to preferentially select which group they join by calling for volunteers for each prompt, and feel free to use 5 minutes from the next (dicussion) sections doing so if you choose to (since this could be viewed as being a part of the "discussion").

> *For each of the prompts, groups should consider the pros and cons of two options, the potential impact of a decision to persue one of the options, and how they take into account how uncertainty influences their thinking about the options.<br><br>*
>
> <details class="details-example"><summary style="color:blue"><u>Further Guidance</u></summary>
>     
> I asked a ChatBot to create a group activity for you that was related to decision-making under uncertainty using probability, and it produced the following questions. 
> 
> *This is a little bit like one of the ideas in the "Afterward" of HW01 asking a ChatBot to suggest and explain some other, perhaps less well-known "unintuitive surprising statistics paradoxes" (besides the "World War 2 Plane" and "Monte Hall" problems)*
>
> </details>

1. **Stock Investment Strategy:** Students are investors trying to maximize their returns in the stock market. They must decide between two investment strategies: "diversified portfolio" or "focused portfolio." Each strategy has different probabilities of success based on market conditions.<br><br>
    
    1. Diversified Portfolio: Spread investments across multiple industries.
    2. Focused Portfolio: Concentrate investments in a few high-potential stocks.<br><br>
        
2. **Healthcare Treatment Decision:** Students are healthcare professionals deciding between two treatment options for a patient's condition. Each treatment has different success rates and potential side effects.<br><br>
    
    1. Treatment A: High success rate but moderate side effects.
    2. Treatment B: Lower success rate but minimal side effects.<br><br>
        
3. **Sports Team Strategy:** Students are coaches of a sports team planning their game strategy. They must decide between two tactics: "offensive strategy" or "defensive strategy." Each strategy has different probabilities of winning based on the opponent's strengths and weaknesses.<br><br>
    
    1. Offensive Strategy: Focus on scoring goals/points aggressively.
    2. Defensive Strategy: Prioritize defense to prevent the opponent from scoring.<br><br>
        
4. **Career Path Decision:** Students are recent graduates deciding between two career paths: "corporate job" or "entrepreneurship." Each path has different probabilities of success and factors to consider, such as job security, income potential, and work-life balance.<br><br>
    
    1. Corporate Job: Stable income but limited growth opportunities.
    2. Entrepreneurship: Higher potential for success but greater risk and uncertainty.<br><br>
        
5. **Environmental Conservation Strategy:** Students are environmental activists advocating for conservation efforts in a wildlife reserve. They must decide between two conservation strategies: "habitat preservation" or "species reintroduction." Each strategy has different probabilities of achieving long-term sustainability for the ecosystem.<br><br>
    
    1. Habitat Preservation: Protect existing habitats from human encroachment.
    2. Species Reintroduction: Reintroduce endangered species to restore ecological balance.


#### 2. **[15 to 20 of the 40 minutes]** Each group plans and prepares a brief (approximately 3 minute) summary (a) introducing their problem context and (b) outlining their decision and the rationale behind it. The group presentations should address

1. the expected outcomes of their decision
2. the risks involved
3. and why they believe their choice is the best in light of their characterization of the degree uncertainty present in their context 

#### 3. **[15 to 20 of the 40 minutes]** Each group gives their (approximately 3 minute) planned presentation. If time permits, engaging in some (students or TA) Q&A seeking clarification or challenging group decisions would be ideal.

Groups who manage to plan a presentation where multiple group members actively part of the presentation should be awarded "gold stars", figuratively, of course, since actualy gold, or stars, of gold star stickers are unfortunately momentarily in short supply
