## Project Overview

Autonomous vehicles operating in urban environments must not only detect pedestrians and cyclists but also anticipate where they are likely to move next. Simply reacting to their current position is not enough for safe navigation—systems need to predict future movement in advance.

This project focuses on **intent and trajectory prediction**, where the goal is to forecast the future path of pedestrians and cyclists based on their recent motion. Given **2 seconds of past movement data (positions/velocity)**, the model predicts their **future positions over the next 3 seconds**.

The challenge lies in the fact that human movement is not always predictable:

- **Multiple possible futures**: A person can turn, stop, or continue straight  
- **Social behavior**: People adjust their movement based on others around them  
- **Temporal patterns**: Motion changes over time and must be understood sequentially  

To address this, the system is designed to:

- Learn movement patterns from **past trajectory data**  
- Consider **interactions between nearby agents**  
- Generate **multiple possible future paths (multi-modal prediction)**  
- Infer likely behavior (**intent**) from observed motion  

The final outcome is a model that takes past movement as input and predicts several realistic future trajectories, helping autonomous systems make **safer and more proactive decisions**.

---

## Model Architecture


---

## Dataset Used


---

## Setup & Installation Instructions


---

## How to Run the Code


---

## Example Outputs / Results