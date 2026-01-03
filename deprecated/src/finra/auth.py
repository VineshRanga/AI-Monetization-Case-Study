"""FINRA OAuth2 authentication."""
import base64
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"


class FinraAuthError(Exception):
    """Exception raised for FINRA authentication errors."""
    pass


def get_access_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> str:
    """
    Get FINRA OAuth2 access token using client credentials flow.
    
    Args:
        client_id: FINRA client ID (defaults to FINRA_CLIENT_ID env var)
        client_secret: FINRA client secret (defaults to FINRA_CLIENT_SECRET env var)
    
    Returns:
        Access token string
    
    Raises:
        FinraAuthError: If authentication fails
    """
    if client_id is None:
        client_id = os.getenv("FINRA_CLIENT_ID")
    if client_secret is None:
        client_secret = os.getenv("FINRA_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise FinraAuthError(
            "FINRA_CLIENT_ID and FINRA_CLIENT_SECRET must be set in environment"
        )
    
    # Build Basic Auth header: base64(client_id:client_secret)
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    auth_header = f"Basic {encoded_credentials}"
    
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    try:
        response = requests.post(
            TOKEN_URL,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise FinraAuthError(f"No access_token in response: {token_data}")
        
        return access_token
    
    except requests.exceptions.RequestException as e:
        raise FinraAuthError(f"Failed to get access token: {e}") from e

