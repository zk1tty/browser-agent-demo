import asyncio
import os
from playwright.async_api import async_playwright
from typing import Optional

async def upload_file_to_chatgpt(
    file_path: str,
    headless: bool = False,
    browser_path: Optional[str] = None
) -> bool:
    """
    Upload a file to ChatGPT using Playwright.
    
    Args:
        file_path (str): Path to the file to upload
        headless (bool): Whether to run browser in headless mode
        browser_path (Optional[str]): Path to Chrome executable
        
    Returns:
        bool: True if upload was successful, False otherwise
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(
            headless=headless,
            executable_path=browser_path
        )
        
        # Create new context and page
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Navigate to ChatGPT
            await page.goto("https://chat.openai.com/")
            
            # Wait for the file upload button to be visible
            upload_button = await page.wait_for_selector('input[type="file"]')
            
            # Upload the file
            await upload_button.set_input_files(file_path)
            
            # Wait for upload to complete (you might need to adjust this based on the actual UI)
            await page.wait_for_timeout(2000)  # Wait 2 seconds for upload
            
            return True
            
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False
            
        finally:
            await browser.close()

# Example usage:
async def main():
    success = await upload_file_to_chatgpt("test.png")
    print(f"Upload successful: {success}") 

if __name__ == "__main__":
    asyncio.run(main()) 