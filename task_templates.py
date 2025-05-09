"""
Task templates for Run Agent tab.
Each template is a string containing step-by-step instructions for the AI agent.
"""

TASK_TEMPLATES = {
    "post_to_x": """go to 'X.com', then:
1. Wait 2 seconds after the page loads
2. Click the 'Post' button
3. Wait 1 second after the post dialog opens
4. In the text input field, type the text VERY SLOWLY and CAREFULLY:
   '  hello world, I'm https://rebrowse.me testing...'
5. After typing, verify EACH CHARACTER:
   - Has 'https://' not 'http://'
   - Has 'rebrowse.me' not 'rebrowse.m'
6. If ANY character is wrong, clear the field completely and type again
7. Only click 'Post' when you've verified every character is correct
8. If you see a rate limit error, wait 30 seconds before retrying""",

    "scrape_website": """go to 'example.com', then:
1. Wait for the page to load completely
2. Extract all product information:
   - Product names
   - Prices
   - Descriptions
3. Save the data in a structured format
4. Verify all data is captured correctly""",

    "heygen_video": """go to 'heygen.ai', then:
1. Wait for the page to load
2. Click the 'Create Video' button
3. Wait for the video creation to complete
4. Verify the video is created successfully
5. Save the video to your computer""",
    
    "custom_task": "Enter your custom task here..."
} 