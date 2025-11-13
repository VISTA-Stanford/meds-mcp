"""
VPN connectivity check utilities for Stanford services.
"""

import socket
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import requests, but make it optional
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def check_stanford_vpn(
    timeout: float = 3.0,
    test_endpoints: Optional[list] = None
) -> Tuple[bool, str]:
    """
    Check if connected to Stanford VPN by testing connectivity to VPN-only services.
    
    Note: Just being able to connect to port 443 doesn't mean VPN is active.
    This function checks for actual VPN-only services.
    
    Args:
        timeout: Connection timeout in seconds
        test_endpoints: Optional list of (host, port) tuples to test.
                      Defaults to VPN-only Stanford internal services.
    
    Returns:
        Tuple of (is_connected: bool, message: str)
    """
    if test_endpoints is None:
        # VPN-only endpoints that should NOT be reachable without VPN
        # These are internal-only services
        test_endpoints = [
            ("apim.stanfordhealthcare.org", 443),  # Stanford Healthcare APIM service (VPN required)
        ]
    
    successful_connections = []
    failed_connections = []
    
    for host, port in test_endpoints:
        try:
            # Try to resolve DNS first
            try:
                ip = socket.gethostbyname(host)
                # Check if IP is in Stanford's internal ranges (171.64.x.x, 171.65.x.x, etc.)
                # This is a better indicator than just port connectivity
                ip_parts = ip.split('.')
                if len(ip_parts) == 4:
                    first_octet = int(ip_parts[0])
                    second_octet = int(ip_parts[1])
                    # Stanford internal IP ranges
                    if first_octet == 171 and second_octet in [64, 65, 66, 67]:
                        # This is a Stanford internal IP - good sign
                        logger.debug(f"{host} resolves to Stanford internal IP: {ip}")
            except socket.gaierror:
                failed_connections.append(f"{host}:{port} (DNS resolution failed)")
                continue
            
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                successful_connections.append(f"{host}:{port}")
            else:
                failed_connections.append(f"{host}:{port} (connection refused)")
                
        except Exception as e:
            failed_connections.append(f"{host}:{port} (error: {str(e)})")
    
    # Just being able to connect doesn't mean VPN is active
    # We need to actually verify the service responds correctly
    if successful_connections:
        # Return uncertain - need HTTP check to verify
        return False, f"⚠️  Port connectivity detected but VPN status uncertain. Use HTTP check for verification."
    else:
        return False, f"❌ Cannot reach Stanford VPN-only services. Failed: {', '.join(failed_connections)}"


def check_vpn_with_http_test() -> Tuple[bool, str]:
    """
    Check VPN connectivity by attempting HTTP request to Stanford internal service.
    
    This is more reliable than socket checks because it verifies the service
    actually responds, not just that the port is open.
    
    Returns:
        Tuple of (is_connected: bool, message: str)
    """
    if not HAS_REQUESTS:
        return False, "⚠️  HTTP check unavailable (requests library not installed)"
    
    try:
        # Try to reach Stanford Healthcare APIM - this requires VPN access
        # Even if we get 401/403, it means we reached the service (VPN is working)
        # Connection errors/timeouts mean VPN is likely not connected
        response = requests.get(
            "https://apim.stanfordhealthcare.org",
            timeout=5,
            verify=True,
            allow_redirects=False
        )
        # Any response (even 401, 403, 404) means we reached the service = VPN is working
        return True, f"✅ VPN Connected - Stanford Healthcare APIM service is reachable (HTTP {response.status_code})"
    except requests.exceptions.SSLError as e:
        # SSL errors might indicate VPN issues, but could also be cert problems
        return False, f"❌ SSL Error - Cannot verify Stanford certificate: {str(e)}"
    except requests.exceptions.Timeout:
        return False, "❌ Timeout - Cannot reach Stanford APIM service (likely not on VPN)"
    except requests.exceptions.ConnectionError as e:
        # Connection errors typically mean VPN is not connected
        error_str = str(e)
        if "Name or service not known" in error_str or "nodename nor servname" in error_str:
            return False, "❌ DNS resolution failed - Cannot resolve Stanford services (VPN likely not connected)"
        return False, f"❌ Connection Error - Cannot reach Stanford services: {error_str}"
    except Exception as e:
        return False, f"❌ Error checking VPN: {str(e)}"


def check_vpn_combined() -> Tuple[bool, str]:
    """
    Combined VPN check using HTTP method (most reliable).
    
    HTTP check is preferred because it verifies the service actually responds,
    not just that a port is open. Socket checks can give false positives.
    
    Returns:
        Tuple of (is_connected: bool, message: str)
    """
    # HTTP check is more reliable - it verifies actual service response
    if HAS_REQUESTS:
        http_connected, http_msg = check_vpn_with_http_test()
        return http_connected, http_msg
    
    # Fall back to socket check if requests not available
    socket_connected, socket_msg = check_stanford_vpn()
    
    # Socket check alone is not reliable, so we're more conservative
    if socket_connected:
        return False, socket_msg + " (Note: Install 'requests' package for accurate VPN detection)"
    
    return False, socket_msg + " (Note: Install 'requests' package for more accurate HTTP-based VPN detection)"

