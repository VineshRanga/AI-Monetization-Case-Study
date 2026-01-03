"""SEC EDGAR API client with rate limiting and fair access compliance."""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


class SecClient:
    """SEC EDGAR API client with rate limiting."""
    
    def __init__(self):
        user_agent = os.getenv("SEC_USER_AGENT")
        if not user_agent:
            raise ValueError(
                "SEC_USER_AGENT must be set in environment. "
                "Format: 'Your Name your.email@domain.com'"
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        
        # Rate limiter: max 8 requests/sec (conservative)
        self.min_request_interval = 1.0 / 8.0  # 0.125 seconds
        self.last_request_time = 0.0
        
        # Data directory for raw JSON
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "sec"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request_with_retries(
        self,
        url: str,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> requests.Response:
        """Make request with exponential backoff retries."""
        for attempt in range(max_retries):
            self._rate_limit()
            
            try:
                response = self.session.get(url, timeout=30)
                
                # Success
                if response.status_code == 200:
                    return response
                
                # Rate limit - retry with backoff
                if response.status_code == 429:
                    wait_time = backoff_base ** attempt
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    response.raise_for_status()
                
                # Server error - retry
                if 500 <= response.status_code < 600:
                    wait_time = backoff_base ** attempt
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    response.raise_for_status()
                
                # Other errors - raise
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff_base ** attempt
                time.sleep(wait_time)
        
        raise RuntimeError("Max retries exceeded")
    
    def _save_json(self, data: Any, filename: str):
        """Save JSON to data/raw/sec/."""
        filepath = self.data_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_company_tickers_exchange(self) -> Dict[str, Any]:
        """
        Get company tickers exchange mapping.
        
        Returns:
            JSON response with ticker -> CIK mapping
        """
        url = TICKERS_URL
        response = self._request_with_retries(url)
        data = response.json()
        
        # Save raw JSON
        self._save_json(data, "company_tickers_exchange.json")
        
        return data
    
    def get_submissions(self, cik: str) -> Dict[str, Any]:
        """
        Get company submissions (filing index).
        
        Args:
            cik: 10-digit zero-padded CIK
        
        Returns:
            JSON response with filing history
        """
        # Ensure CIK is 10-digit zero-padded
        cik_padded = cik.zfill(10)
        
        url = f"{BASE_URL}/submissions/CIK{cik_padded}.json"
        response = self._request_with_retries(url)
        data = response.json()
        
        # Save raw JSON
        self._save_json(data, f"submissions_CIK{cik_padded}.json")
        
        return data
    
    def get_companyfacts(self, cik: str) -> Dict[str, Any]:
        """
        Get XBRL company facts (all facts).
        
        Args:
            cik: 10-digit zero-padded CIK
        
        Returns:
            JSON response with company facts
        """
        # Ensure CIK is 10-digit zero-padded
        cik_padded = cik.zfill(10)
        
        url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        response = self._request_with_retries(url)
        data = response.json()
        
        # Save raw JSON
        self._save_json(data, f"companyfacts_CIK{cik_padded}.json")
        
        return data


def get_company_tickers_exchange() -> Dict[str, Any]:
    """Convenience function to get ticker mapping."""
    client = SecClient()
    return client.get_company_tickers_exchange()


def get_submissions(cik: str) -> Dict[str, Any]:
    """Convenience function to get submissions."""
    client = SecClient()
    return client.get_submissions(cik)


def get_companyfacts(cik: str) -> Dict[str, Any]:
    """Convenience function to get company facts."""
    client = SecClient()
    return client.get_companyfacts(cik)

