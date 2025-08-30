import httpx
from aider.tools.base_tool import BaseAiderTool

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

class CnnScraper(BaseAiderTool):
    """
    A tool to scrape the top story headlines from www.cnn.com.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "CnnScraper",
                "description": "Scrapes www.cnn.com to get the top story headlines.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def run(self, **kwargs):
        """
        Scrapes www.cnn.com and returns the top story headlines.
        
        :return: A string containing the top headlines, or an error message.
        """
        if not BeautifulSoup:
            return "Error: `beautifulsoup4` is required for this tool. Please install it using: pip install beautifulsoup4"

        url = "https://www.cnn.com"
        try:
            self.coder.io.tool_output(f"Scraping headlines from {url}...")
            
            # Use httpx to get the page content
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }
            response = httpx.get(url, follow_redirects=True, timeout=15, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse the HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find headline elements. This selector might need updating if CNN changes their site structure.
            # It targets text within spans that have a specific class, inside links that are articles.
            headline_elements = soup.select('a[data-link-type="article"] span.container__headline-text')
            
            headlines = []
            for el in headline_elements:
                text = el.get_text(strip=True)
                if text:
                    headlines.append(text)
            
            if not headlines:
                # Fallback for simpler structures if the primary selector fails
                self.coder.io.tool_output("Primary selector failed, trying fallback...")
                headline_elements = soup.find_all(['h1', 'h2', 'h3'])
                for el in headline_elements:
                    # Check if the element is inside an anchor tag
                    if el.find_parent('a'):
                        text = el.get_text(strip=True)
                        if text and len(text) > 20: # filter out short titles
                            headlines.append(text)

            # Remove duplicates while preserving order
            unique_headlines = list(dict.fromkeys(headlines))

            if not unique_headlines:
                return "Could not find any headlines on CNN. The website structure may have changed."
            
            # Format the output, returning up to the top 15 headlines
            output = "Top story headlines from CNN:\n"
            for i, headline in enumerate(unique_headlines[:15], 1):
                output += f"{i}. {headline}\n"
                
            return output.strip()

        except ImportError:
            return "Error: `httpx` is required for this tool. Please install it using: pip install httpx"
        except httpx.RequestError as e:
            return f"Error fetching URL {url}: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

