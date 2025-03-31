from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time

class BilibiliController:
    def __init__(self):
        # Initialize Chrome browser with custom options
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")  # Start with maximized window
        
        # Initialize Chrome driver with webdriver-manager
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.get("https://www.bilibili.com/video/BV1ps411B7EX/?spm_id_from=333.337.search-card.all.click&vd_source=fed6f92e920819ddbb3f7598f68edce6")
        
        # Give the user some time to log in (if needed)
        print("Please log in to Bilibili within 30 seconds (if needed)")
        time.sleep(30)
        
    def next_video(self):
        """Play the next video"""
        try:
            # Try different selectors, as Bilibili may have different layouts
            selectors = [
                ".video-page-card-small", 
                ".recommend-list-v1 .video-card",
                ".card-box .video-card",
                ".archive-list .video-card",
                ".video-card",  # More generic selector
                ".rec-list .video-card",
                "a.title"  # Title links that might be videos
            ]
            
            for selector in selectors:
                next_videos = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if next_videos and len(next_videos) > 0:
                    next_videos[0].click()
                    time.sleep(1)  # Wait for the page to load
                    return True
            
            # If we can't find recommended videos, try using JavaScript to navigate
            self.driver.execute_script("window.location.href = document.querySelector('a[href*=\"video\"]').href")
            return True
            
        except Exception as e:
            print(f"Error switching videos: {e}")
            return False
    
    def volume_up(self):
        """Increase volume"""
        try:
            # Focus on the player itself
            video_player = self.driver.find_element(By.CSS_SELECTOR, ".bilibili-player-video video")
            if video_player:
                video_player.click()
            
            # Use ActionChains to send the up arrow key
            ActionChains(self.driver).send_keys(Keys.ARROW_UP).perform()
            return True
        except Exception as e:
            print(f"Error adjusting volume: {e}")
            
            # Alternative method: try using JavaScript to adjust volume
            try:
                # Many video players have a volume property that can be changed via JavaScript
                self.driver.execute_script("""
                    var video = document.querySelector('video');
                    if (video) {
                        video.volume = Math.min(1.0, video.volume + 0.1);
                    }
                """)
                return True
            except:
                return False
    
    def close(self):
        """Close the browser"""
        try:
            self.driver.quit()
        except:
            pass