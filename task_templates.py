"""
Task templates for Run Agent tab.
Each template is a string containing step-by-step instructions for the AI agent.
"""

TASK_TEMPLATES = {
    "post_to_x": """go to 'X.com', then:
1. In the text input field, type the text:
   'hello world, I'm rebrowse.me'
2. click 'Post'
3. confirm that you can see the tweet posted on the timeline.""",

    "heygen_video": """go to 'heygen.ai', then:
1. Wait for the page to load
2. Click the 'Create Video' button
3. Wait for the video creation to complete
4. Verify the video is created successfully
5. Save the video to your computer""",
    
    "custom_task": "Enter your custom task here..."
} 