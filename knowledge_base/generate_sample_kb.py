"""
Generate a sample TechCorp Customer Support Knowledge Base PDF.
This creates a realistic ~10-page PDF for the RAG system demo.

Usage:
    python knowledge_base/generate_sample_kb.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF


class TechCorpKBPDF(FPDF):
    """Custom PDF class for TechCorp knowledge base."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "TechCorp Customer Support Knowledge Base | Confidential", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(51, 51, 153)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(51, 51, 153)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(3)


def generate_kb():
    """Generate the complete knowledge base PDF."""
    pdf = TechCorpKBPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ─── Page 1: Cover / Overview ────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(51, 51, 153)
    pdf.ln(40)
    pdf.cell(0, 15, "TechCorp", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 18)
    pdf.cell(0, 12, "Customer Support Knowledge Base", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Version 2.1 | Last Updated: April 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Internal Use Only", align="C", new_x="LMARGIN", new_y="NEXT")

    # ─── Page 2: Company & Products Overview ─────────────────
    pdf.add_page()
    pdf.chapter_title("1. Company & Products Overview")

    pdf.body_text(
        "TechCorp is a leading technology company specializing in smart home devices, "
        "cloud services, and cybersecurity solutions. Founded in 2018, TechCorp serves "
        "over 2 million customers worldwide with three flagship products."
    )

    pdf.section_title("1.1 SmartHome Hub (Model SH-500)")
    pdf.body_text(
        "The SmartHome Hub is TechCorp's flagship smart home controller. It connects up to "
        "100 IoT devices including lights, thermostats, cameras, and door locks. The SH-500 "
        "features a 7-inch touchscreen, built-in Alexa and Google Assistant, and supports "
        "WiFi 6, Zigbee, Z-Wave, and Bluetooth 5.0 protocols. Price: $199.99 (Standard), "
        "$299.99 (Pro with extended range)."
    )
    pdf.body_text(
        "Key Features: Voice control in 12 languages, automated routines and scenes, "
        "energy monitoring dashboard, integration with 500+ third-party devices, "
        "local processing for privacy-sensitive commands, OTA firmware updates."
    )

    pdf.section_title("1.2 CloudSync Pro")
    pdf.body_text(
        "CloudSync Pro is TechCorp's enterprise-grade cloud storage and collaboration platform. "
        "It offers end-to-end encrypted file storage, real-time document collaboration, "
        "and automated backup solutions. Plans: Basic (100GB, $4.99/month), Professional "
        "(1TB, $12.99/month), Enterprise (Unlimited, $29.99/user/month). All plans include "
        "a 30-day free trial."
    )
    pdf.body_text(
        "Features: AES-256 encryption at rest and in transit, version history up to 180 days, "
        "file sharing with granular permissions, desktop sync client for Windows/Mac/Linux, "
        "mobile apps for iOS and Android, API access for enterprise integrations, "
        "compliance with SOC2, HIPAA, and GDPR standards."
    )

    pdf.section_title("1.3 SecureVPN")
    pdf.body_text(
        "SecureVPN is TechCorp's consumer and business VPN service providing private and "
        "secure internet access. It operates 3,000+ servers across 60 countries with a "
        "strict no-logs policy. Plans: Monthly ($11.99), Annual ($5.99/month billed yearly), "
        "Family (up to 6 devices, $8.99/month billed yearly)."
    )
    pdf.body_text(
        "Features: WireGuard and OpenVPN protocols, kill switch, split tunneling, "
        "ad and malware blocking, dedicated IP option ($3/month extra), "
        "router-level VPN support, 24/7 live chat support, 30-day money-back guarantee."
    )

    # ─── Page 3-4: Setup & Installation ──────────────────────
    pdf.add_page()
    pdf.chapter_title("2. Setup & Installation Guides")

    pdf.section_title("2.1 SmartHome Hub Setup")
    pdf.body_text(
        "Step 1: Unbox the SmartHome Hub and plug it into a power outlet using the included "
        "USB-C power adapter. Place it in a central location in your home for optimal WiFi "
        "and Zigbee range.\n\n"
        "Step 2: The hub will power on automatically. Wait for the LED ring to turn solid blue "
        "(approximately 30 seconds). If the LED is red, the device needs a firmware update - "
        "connect to WiFi first and it will update automatically.\n\n"
        "Step 3: Download the TechCorp Home app from the App Store or Google Play. Create an "
        "account or sign in with your existing TechCorp ID.\n\n"
        "Step 4: In the app, tap 'Add Device' > 'SmartHome Hub'. The app will scan for nearby "
        "hubs. Select your hub (identified by the serial number on the bottom of the device).\n\n"
        "Step 5: Connect the hub to your WiFi network. Enter your WiFi password when prompted. "
        "The hub supports both 2.4GHz and 5GHz networks.\n\n"
        "Step 6: Follow the on-screen wizard to set up voice assistants, create rooms, "
        "and add your first smart devices."
    )

    pdf.section_title("2.2 CloudSync Pro Setup")
    pdf.body_text(
        "Step 1: Visit cloudsync.techcorp.com and click 'Start Free Trial' or sign in.\n\n"
        "Step 2: Choose your plan (Basic, Professional, or Enterprise). All plans start with "
        "a 30-day free trial. No credit card required for trial.\n\n"
        "Step 3: Download the desktop sync client from your dashboard. Available for Windows 10+, "
        "macOS 12+, and Ubuntu 20.04+.\n\n"
        "Step 4: Install and launch the client. Sign in with your TechCorp ID. Choose which "
        "folders to sync (default: Documents, Desktop, Pictures).\n\n"
        "Step 5: The client will begin initial sync. Large uploads are handled in the background "
        "with bandwidth throttling options available in Settings > Sync > Bandwidth."
    )

    pdf.section_title("2.3 SecureVPN Setup")
    pdf.body_text(
        "Step 1: Purchase a SecureVPN plan at securevpn.techcorp.com.\n\n"
        "Step 2: Download the SecureVPN app for your device (Windows, Mac, iOS, Android, Linux).\n\n"
        "Step 3: Install and open the app. Sign in with your TechCorp credentials.\n\n"
        "Step 4: Click 'Quick Connect' to automatically connect to the fastest server, or "
        "choose a specific country from the server list.\n\n"
        "Step 5: Enable the kill switch (Settings > Security > Kill Switch) to prevent data "
        "leaks if the VPN connection drops.\n\n"
        "Step 6: For router setup, go to Settings > Advanced > Router Configuration and follow "
        "the guides for your specific router model."
    )

    # ─── Page 5-6: Troubleshooting ───────────────────────────
    pdf.add_page()
    pdf.chapter_title("3. Troubleshooting Guide")

    pdf.section_title("3.1 SmartHome Hub Issues")

    pdf.body_text(
        "Problem: Hub won't connect to WiFi\n"
        "Solution: 1) Ensure your router is broadcasting on 2.4GHz (some devices struggle with "
        "5GHz-only networks). 2) Move the hub closer to your router during setup. 3) Restart "
        "your router and the hub. 4) Factory reset the hub by pressing and holding the reset "
        "button on the bottom for 10 seconds until the LED flashes red three times. 5) If the "
        "issue persists, check if your router has MAC filtering enabled and whitelist the hub's "
        "MAC address (found on the bottom label)."
    )
    pdf.body_text(
        "Problem: Smart devices not responding\n"
        "Solution: 1) Check if the device is powered on and within range. 2) In the TechCorp "
        "Home app, go to the device and tap 'Reconnect'. 3) For Zigbee devices, ensure they are "
        "within 30 feet of the hub or a Zigbee repeater. 4) Remove the device from the app and "
        "re-add it. 5) Check for firmware updates for both the hub and the device."
    )
    pdf.body_text(
        "Problem: Hub is slow or unresponsive\n"
        "Solution: 1) Check your internet speed (minimum 10 Mbps recommended). 2) Reduce the "
        "number of connected devices if over 75. 3) Clear the hub's cache in Settings > System > "
        "Clear Cache. 4) Perform a soft reset by unplugging for 30 seconds. 5) Check for "
        "firmware updates in Settings > System > Firmware Update."
    )

    pdf.section_title("3.2 CloudSync Pro Issues")
    pdf.body_text(
        "Problem: Files not syncing\n"
        "Solution: 1) Check your internet connection. 2) Ensure the sync client is running "
        "(look for the cloud icon in your system tray). 3) Check available storage in your plan "
        "(Dashboard > Storage). 4) Verify the file isn't locked by another application. "
        "5) Check for filename issues - CloudSync doesn't support files with special characters "
        "like <, >, :, \", |, ?, * in names. 6) Restart the sync client."
    )
    pdf.body_text(
        "Problem: Slow upload speeds\n"
        "Solution: 1) Check your internet upload speed at speedtest.techcorp.com. 2) Go to "
        "Settings > Sync > Bandwidth and ensure 'Unlimited' is selected. 3) Pause other "
        "bandwidth-heavy applications. 4) For large initial syncs (>50GB), consider using "
        "the 'Bulk Upload' feature in the web interface which is optimized for large transfers."
    )

    pdf.section_title("3.3 SecureVPN Issues")
    pdf.body_text(
        "Problem: Cannot connect to VPN\n"
        "Solution: 1) Try a different server location. 2) Switch protocol from WireGuard to "
        "OpenVPN (Settings > Protocol). 3) Check if your firewall or antivirus is blocking the "
        "VPN. 4) If on a restricted network (hotel, office), enable 'Stealth Mode' in Settings > "
        "Advanced. 5) Reinstall the VPN app if issues persist."
    )
    pdf.body_text(
        "Problem: Slow VPN speeds\n"
        "Solution: 1) Connect to the nearest server geographically. 2) Switch to WireGuard "
        "protocol for best performance. 3) Disable ad/malware blocking temporarily. "
        "4) Check if split tunneling is routing unnecessary traffic through the VPN. "
        "5) Try a different server in the same country."
    )

    # ─── Page 7: Billing & Subscription ──────────────────────
    pdf.add_page()
    pdf.chapter_title("4. Billing & Subscription FAQ")

    pdf.body_text(
        "Q: How do I view my current subscription?\n"
        "A: Log in to your TechCorp account at account.techcorp.com. Go to 'Subscriptions' "
        "to see all active plans, billing dates, and payment methods."
    )
    pdf.body_text(
        "Q: How do I change my subscription plan?\n"
        "A: Go to account.techcorp.com > Subscriptions > select the product > 'Change Plan'. "
        "Upgrades take effect immediately with prorated billing. Downgrades take effect at the "
        "end of the current billing cycle."
    )
    pdf.body_text(
        "Q: What payment methods are accepted?\n"
        "A: TechCorp accepts Visa, Mastercard, American Express, PayPal, and bank transfers "
        "(Enterprise plans only). Cryptocurrency payments are not currently supported."
    )
    pdf.body_text(
        "Q: How do I update my payment method?\n"
        "A: Go to account.techcorp.com > Payment Methods > 'Add New' or 'Edit'. You can have "
        "up to 5 payment methods on file. The primary method is charged for all subscriptions."
    )
    pdf.body_text(
        "Q: Why was I charged twice?\n"
        "A: Double charges can occur if: 1) You have multiple active subscriptions (check "
        "Subscriptions page). 2) A pending authorization from a failed payment attempt was "
        "retried. Pending charges typically clear within 3-5 business days. 3) You upgraded "
        "your plan mid-cycle (prorated charge + next cycle charge). Contact billing support "
        "if the double charge persists beyond 5 business days."
    )
    pdf.body_text(
        "Q: How do I cancel auto-renewal?\n"
        "A: Go to account.techcorp.com > Subscriptions > select product > 'Cancel Auto-Renewal'. "
        "Your service will continue until the end of the current billing period. You can "
        "reactivate anytime before the period ends without losing data."
    )

    # ─── Page 8: Refund & Cancellation Policy ────────────────
    pdf.add_page()
    pdf.chapter_title("5. Refund & Cancellation Policy")

    pdf.section_title("5.1 Software & Services Refund Policy")
    pdf.body_text(
        "All TechCorp software subscriptions (CloudSync Pro, SecureVPN) come with a 30-day "
        "money-back guarantee from the date of initial purchase. To request a refund:\n\n"
        "1. Log in to account.techcorp.com\n"
        "2. Go to Subscriptions > select the product\n"
        "3. Click 'Request Refund'\n"
        "4. Select a reason for the refund\n"
        "5. Submit the request\n\n"
        "Refund processing times:\n"
        "- Credit/Debit Card: 5-10 business days\n"
        "- PayPal: 3-5 business days\n"
        "- Bank Transfer: 10-15 business days\n\n"
        "Refunds after the 30-day period are handled on a case-by-case basis. Prorated refunds "
        "may be issued for annual plans cancelled within the first 90 days."
    )

    pdf.section_title("5.2 Hardware Refund Policy")
    pdf.body_text(
        "SmartHome Hub devices can be returned within 60 days of purchase for a full refund, "
        "provided the device is in its original packaging and in working condition. "
        "Damaged or modified devices are not eligible for refund but may qualify for warranty "
        "replacement. To initiate a hardware return:\n\n"
        "1. Contact support at returns.techcorp.com\n"
        "2. Provide your order number and reason for return\n"
        "3. Receive a prepaid return shipping label via email\n"
        "4. Ship the device within 14 days of receiving the label\n"
        "5. Refund is processed within 5 business days of receiving the device"
    )

    pdf.section_title("5.3 Cancellation Policy")
    pdf.body_text(
        "Cancellation of any TechCorp subscription can be done at any time through your account "
        "settings. Key points:\n\n"
        "- Monthly plans: Cancellation takes effect at the end of the current month\n"
        "- Annual plans: Cancellation takes effect at the end of the current annual period\n"
        "- Enterprise plans: 30-day written notice required\n"
        "- Data retention: Your data is retained for 90 days after cancellation, then permanently "
        "deleted. You can download your data at any time during this period.\n"
        "- Reactivation: You can reactivate within 90 days without losing data"
    )

    # ─── Page 9: Shipping & Delivery ─────────────────────────
    pdf.add_page()
    pdf.chapter_title("6. Shipping & Delivery")

    pdf.body_text(
        "TechCorp ships hardware products (SmartHome Hub) to customers in 35+ countries. "
        "All orders are processed within 1-2 business days."
    )

    pdf.section_title("6.1 Shipping Options (United States)")
    pdf.body_text(
        "Standard Shipping: 5-7 business days, FREE for orders over $50, $5.99 otherwise.\n"
        "Express Shipping: 2-3 business days, $12.99.\n"
        "Next-Day Shipping: 1 business day (order by 2 PM EST), $24.99.\n"
        "All shipments include tracking via UPS or FedEx."
    )

    pdf.section_title("6.2 International Shipping")
    pdf.body_text(
        "International shipping is available to Canada, UK, EU, Australia, and Japan. "
        "Delivery times range from 7-14 business days. Shipping costs start at $19.99. "
        "Import duties and taxes are the responsibility of the customer and are not included "
        "in the shipping cost. TechCorp uses DHL for international shipments."
    )

    pdf.section_title("6.3 Order Tracking")
    pdf.body_text(
        "Track your order at orders.techcorp.com using your order number or TechCorp account. "
        "You will receive email notifications at each stage: order confirmed, shipped, "
        "out for delivery, and delivered. SMS notifications can be enabled in your account "
        "settings under Notifications > Order Updates."
    )

    # ─── Page 10: Warranty & Escalation ──────────────────────
    pdf.add_page()
    pdf.chapter_title("7. Warranty & Escalation Procedures")

    pdf.section_title("7.1 Hardware Warranty")
    pdf.body_text(
        "All SmartHome Hub devices come with a 2-year limited manufacturer's warranty covering "
        "defects in materials and workmanship. The warranty does NOT cover: physical damage, "
        "water damage, unauthorized modifications, or damage from power surges. "
        "Extended warranty (additional 2 years) is available for $39.99 at time of purchase "
        "or within 30 days. To make a warranty claim, contact warranty.techcorp.com with your "
        "serial number and description of the issue."
    )

    pdf.section_title("7.2 Support Tiers")
    pdf.body_text(
        "Tier 1 - Self-Service: Knowledge base articles, FAQs, and community forums at "
        "support.techcorp.com. Available 24/7.\n\n"
        "Tier 2 - AI Assistant: Our AI-powered support chatbot available 24/7 for instant "
        "answers to common questions.\n\n"
        "Tier 3 - Email Support: For issues the AI cannot resolve. Response time: 24 hours "
        "(Standard), 4 hours (Professional), 1 hour (Enterprise). Email: support@techcorp.com.\n\n"
        "Tier 4 - Live Chat/Phone: Available Monday-Friday, 9 AM - 6 PM EST. Phone: "
        "1-800-TECHCORP. Live chat available at support.techcorp.com."
    )

    pdf.section_title("7.3 Escalation Procedures")
    pdf.body_text(
        "If your issue is not resolved through standard support channels, you may request "
        "escalation. Escalation criteria:\n\n"
        "- Issue unresolved after 48 hours\n"
        "- Service outage affecting multiple features\n"
        "- Security concern or data breach suspicion\n"
        "- Billing dispute exceeding $100\n"
        "- Repeated issue (3+ occurrences)\n\n"
        "Escalation path: Support Agent > Team Lead > Support Manager > VP of Customer Success.\n"
        "Escalated cases receive a dedicated case manager and are tracked to resolution with "
        "regular updates every 24 hours.\n\n"
        "For urgent security issues, use our emergency hotline: 1-800-TECH-911 (24/7)."
    )

    pdf.section_title("7.4 Service Level Agreement (SLA)")
    pdf.body_text(
        "TechCorp commits to the following SLAs for cloud services:\n\n"
        "- CloudSync Pro: 99.9% uptime guarantee\n"
        "- SecureVPN: 99.5% uptime guarantee\n"
        "- Scheduled maintenance: Announced 48 hours in advance via email and status page\n"
        "- Unscheduled downtime compensation: 10x credit for affected time period\n"
        "- Status page: status.techcorp.com (real-time service health)\n\n"
        "SLA credits are applied automatically to your next billing cycle. Enterprise customers "
        "with custom SLAs should refer to their service agreement."
    )

    # ─── Save PDF ────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "techcorp_support_kb.pdf")
    pdf.output(output_path)
    print(f"[OK] Knowledge base PDF generated: {output_path}")
    print(f"     Pages: {pdf.page_no()}")
    return output_path


if __name__ == "__main__":
    generate_kb()
