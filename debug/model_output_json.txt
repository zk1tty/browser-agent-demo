## Output format
{{
 "current_state": {{
   "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Shortly state why/why not.",
   "important_contents": "Output important contents closely related to user\'s instruction on the current page. If there is, please output the contents. If not, please output empty string ''.",
   "thought": "Think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. If your output of evaluation_previous_goal is 'Failed', please reflect and output your reflection here.",
   "next_goal": "Please generate a brief natural language description for the goal of your next actions based on your thought."
 }},
  "action": [
   {{"one_action_name": {{// action-specific parameter}}}}, // ... more actions in sequence
 ]
}}

## Model output example: case Airbnb

        {
            "current_state": {
            "evaluation_previous_goal": "Failed - The previous action clicked 'About this result' instead of the first search result link.",
            "important_contents": "",
            "thought": "I need to click on the first search result link for 'Airbnb' to proceed to their website.",
            "next_goal": "Click the link for the first search result to go to Airbnb's website."
        },
        "action": [
          {
            "click_element": {
              "index": 18
            }
          }
        ]
      },
      "result": [
        {
          "is_done": false,
          "extracted_content": "\ud83d\uddb1\ufe0f  Clicked button with index 18: Airbnb | Vacation rentals, cabins, beach houses, & more",
          "include_in_memory": true
        }
      ],
      "state": {
        "tabs": [
          {
            "page_id": 0,
            "url": "https://www.google.com/search?q=Airbnb&sca_esv=4f7b8728d565568a&source=hp&ei=JAD0Z9D3JaOf4-EPkpqi8QU&iflsig=ACkRmUkAAAAAZ_QOND2JMcRqrTpGkSd2TrzUJkGQF9uK&ved=0ahUKEwjQmbu5rsaMAxWjzzgGHRKNKF4Q4dUDCA4&oq=Airbnb&gs_lp=Egdnd3Mtd2l6IgZBaXJibmIyFBAuGIAEGLEDGNEDGIMBGMcBGMkDMgsQABiABBixAxiSAzILEAAYgAQYkgMYigUyCBAAGIAEGLEDMggQABiABBixAzIFEAAYgAQyCBAAGIAEGLEDMgUQABiABDIFEAAYgAQyCBAAGIAEGLEDSKI4UABYAHAAeACQAQCYASygASyqAQExuAEMyAEA-AEBmAIBoAIwmAMAkgcBMaAHwgeyBwExuAcw&sclient=gws-wiz&sei=MgD0Z-nrMcXt4-EPheOsuQg",
            "title": "Airbnb - Google Search"
          }
        ],
        "screenshot": "",
        "interacted_element": [
          {
            "tag_name": "a",
            "xpath": "html/body/div[3]/div/div[11]/div/div[2]/div[2]/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/span/a",
            "highlight_index": 18,
            "entire_parent_branch_path": ["div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "div", "span", "a"],
            "attributes": {
              "jsname": "UWckNb",
              "class": "zReHs",
              "href": "https://www.airbnb.com/",
              "data-ved": "2ahUKEwjmtafArsaMAxWOzjgGHbOjBzMQFnoECAoQAQ",
              "ping": "/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.airbnb.com/&ved=2ahUKEwjmtafArsaMAxWOzjgGHbOjBzMQFnoECAoQAQ"
            },
            "shadow_root": false,
            "css_selector": "html > body > div:nth-of-type(3) > div > div:nth-of-type(11) > div > div:nth-of-type(2) > div:nth-of-type(2) > div > div > div > div > div > div > div > div > div > div > div > div:nth-of-type(2) > div > div > span > a.zReHs[href=\"https://www.airbnb.com/\"]",
            "page_coordinates": null,
            "viewport_coordinates": null,
            "viewport_info": null
          }
        ],
        "url": "https://www.google.com/search?q=Airbnb&sca_esv=4f7b8728d565568a&source=hp&ei=JAD0Z9D3JaOf4-EPkpqi8QU&iflsig=ACkRmUkAAAAAZ_QOND2JMcRqrTpGkSd2TrzUJkGQF9uK&ved=0ahUKEwjQmbu5rsaMAxWjzzgGHRKNKF4Q4dUDCA4&oq=Airbnb&gs_lp=Egdnd3Mtd2l6IgZBaXJibmIyFBAuGIAEGLEDGNEDGIMBGMcBGMkDMgsQABiABBixAxiSAzILEAAYgAQYkgMYigUyCBAAGIAEGLEDMggQABiABBixAzIFEAAYgAQyCBAAGIAEGLEDMgUQABiABDIFEAAYgAQyCBAAGIAEGLEDSKI4UABYAHAAeACQAQCYASygASyqAQExuAEMyAEA-AEBmAIBoAIwmAMAkgcBMaAHwgeyBwExuAcw&sclient=gws-wiz&sei=MgD0Z-nrMcXt4-EPheOsuQg",
        "title": "Airbnb - Google Search"
      },
      "metadata": {
        "step_start_time": 1744044083.6483378,
        "step_end_time": 1744044093.986989,
        "input_tokens": 8345,
        "step_number": 5
      }
