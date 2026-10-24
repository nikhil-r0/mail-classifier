import re
import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from mailsort.config import BASE_DIR, DATA_DIR, TARGET_CATEGORIES
from mailsort.preprocessing import clean_text, combine_subject_and_body

logger = logging.getLogger(__name__)

# Curated reference templates representing authentic domain distributions
# with explicit provenance tracking for reproducibility
STARTER_DATASET: List[Dict[str, str]] = [
    # =========================================================================
    # PRIMARY SAMPLES (Direct work, interpersonal, academic, professional)
    # =========================================================================
    {
        "subject": "Project sync meeting tomorrow",
        "body": "Hi team, let's meet tomorrow at 10 AM to discuss the ML assignment progress and review our pipeline deliverables. Please update your branches.",
        "label": "Primary",
        "source": "academic_collaboration"
    },
    {
        "subject": "Questions regarding Lab 4 evaluation",
        "body": "Dear Professor, I have reviewed the feedback on Lab 4 and wanted to clarify the scoring criteria for the confusion matrix analysis. Could we meet during office hours?",
        "label": "Primary",
        "source": "academic_inquiry"
    },
    {
        "subject": "Draft report for BIC703 submission",
        "body": "Hey Nikhil, I have attached the latest draft of our machine learning assignment report. Please read through Section 4 and let me know if any metrics need adjustment before tomorrow.",
        "label": "Primary",
        "source": "academic_collaboration"
    },
    {
        "subject": "Dinner this Friday?",
        "body": "Hey, it has been a while since we caught up! Are you free this Friday evening around 7 PM to grab dinner downtown?",
        "label": "Primary",
        "source": "personal_correspondence"
    },
    {
        "subject": "Notes from our client call",
        "body": "Hi Sarah, attached are the meeting minutes and action items from today's architectural review. Let me know if I missed any key technical constraints.",
        "label": "Primary",
        "source": "professional_work"
    },
    {
        "subject": "Interview scheduling: Software Engineer role",
        "body": "Hi Alex, thank you for taking the time to speak with our recruiting team. We would like to invite you for a 45-minute technical discussion with our lead engineer next Tuesday.",
        "label": "Primary",
        "source": "professional_recruiting"
    },
    {
        "subject": "Doctor's appointment reminder",
        "body": "Dear patient, this is a reminder of your upcoming consultation with Dr. Sharma on Thursday at 3:30 PM. Please arrive 15 minutes early with your medical records.",
        "label": "Primary",
        "source": "healthcare_service"
    },
    {
        "subject": "Apartment lease renewal terms",
        "body": "Dear tenant, your current 12-month lease is set to expire on November 30. Attached is the renewal agreement outlining the upcoming term options and utility schedule.",
        "label": "Primary",
        "source": "administrative_housing"
    },
    {
        "subject": "Code review requested: pull request #142",
        "body": "Hi Dave, could you please take a look at PR #142 when you get a chance? It implements the sequence truncation and OOV token handling we discussed.",
        "label": "Primary",
        "source": "software_engineering"
    },
    {
        "subject": "Family gathering over the weekend",
        "body": "Hi everyone, Mom wants to know who is coming to the family dinner on Sunday. Please let us know by Thursday so we can finalize the food count.",
        "label": "Primary",
        "source": "personal_family"
    },
    {
        "subject": "Thesis defense schedule confirmation",
        "body": "Dear committee members, the defense has been confirmed for Friday at 2:00 PM in Seminar Hall B. The presentation slides will be shared 24 hours in advance.",
        "label": "Primary",
        "source": "academic_administrative"
    },
    {
        "subject": "Feedback on your design proposal",
        "body": "Hi Karen, I went through the UI wireframes you shared. The layout is clean and intuitive. I left a few comments on slide 4 regarding mobile responsive breakpoints.",
        "label": "Primary",
        "source": "professional_design"
    },
    {
        "subject": "Bug report: Database connection timeout under load",
        "body": "Hi Engineering, we noticed intermittent connection pool exhaustion in the user service during peak traffic. Full stack traces and Grafana snapshots are attached.",
        "label": "Primary",
        "source": "software_engineering"
    },
    {
        "subject": "Dentist follow-up consultation",
        "body": "Hello John, this is Dr. Lee's clinic confirming your follow-up cleaning on Wednesday at 11:00 AM. Please call if you need to reschedule.",
        "label": "Primary",
        "source": "healthcare_service"
    },
    {
        "subject": "Regarding course registration deadline",
        "body": "Dear student, this is a reminder from the Registrar's Office that elective course drop/add requests must be finalized by Friday at 5:00 PM.",
        "label": "Primary",
        "source": "academic_administrative"
    },
    {
        "subject": "Weekend hiking trip logistics",
        "body": "Hey guys, weather looks great for our trail hike this Saturday. Let's carpool from the metro station at 7:30 AM. Don't forget plenty of water and snacks.",
        "label": "Primary",
        "source": "personal_correspondence"
    },
    {
        "subject": "Quarterly financial budget review",
        "body": "Dear team, please find attached the spreadsheet for Q3 department expenditures. Please review line items before our finance sync on Thursday morning.",
        "label": "Primary",
        "source": "professional_work"
    },
    {
        "subject": "Recommendation letter request",
        "body": "Dear Professor Mehta, I am applying for graduate studies in computer science and would be deeply grateful if you would consider writing a letter of recommendation on my behalf.",
        "label": "Primary",
        "source": "academic_inquiry"
    },
    {
        "subject": "Flight itinerary and boarding passes",
        "body": "Hi Mark, here is our travel itinerary for the Bangalore conference. Flight departs at 8:15 AM from Terminal 2. Looking forward to our joint presentation.",
        "label": "Primary",
        "source": "professional_travel"
    },
    {
        "subject": "Birthday party celebration invitation",
        "body": "Hey friends, we are throwing a surprise birthday party for Priya this Saturday at 8 PM at our place. Hope you can make it! Keep it a secret.",
        "label": "Primary",
        "source": "personal_correspondence"
    },
    {
        "subject": "Sprint planning agenda for next week",
        "body": "Hi dev team, the backlog for Sprint 14 is ready. Please take a look at the user stories and estimate story points ahead of our planning session on Monday.",
        "label": "Primary",
        "source": "software_engineering"
    },
    {
        "subject": "Carpool coordination for tomorrow's shift",
        "body": "Hey Kevin, my car is in the repair shop. Would you have room for one more on the morning commute tomorrow? Can chip in for gas.",
        "label": "Primary",
        "source": "personal_correspondence"
    },
    {
        "subject": "Assignment 2 submission confirmation",
        "body": "Dear Student, your submission for Assignment 2 (Natural Language Processing) has been successfully recorded in the grading portal at 11:45 PM.",
        "label": "Primary",
        "source": "academic_administrative"
    },
    {
        "subject": "Contract agreement signature required",
        "body": "Hi James, please review and e-sign the non-disclosure agreement attached below so that we can proceed with onboarding next week.",
        "label": "Primary",
        "source": "professional_legal"
    },
    {
        "subject": "Home internet repair technician visit",
        "body": "Dear customer, a service technician is scheduled to visit your premises on Wednesday between 2 PM and 4 PM to inspect your fiber modem line.",
        "label": "Primary",
        "source": "administrative_utilities"
    },
    {
        "subject": "Quick sync regarding performance review",
        "body": "Hi Elena, let's schedule 30 minutes this week to go over your mid-year goals and career progression milestones. Let me know your available slots.",
        "label": "Primary",
        "source": "professional_management"
    },
    {
        "subject": "Library book return reminder",
        "body": "Dear member, the books 'Pattern Recognition and Machine Learning' and 'Speech and Language Processing' are due for return by Monday.",
        "label": "Primary",
        "source": "academic_library"
    },
    {
        "subject": "Thank you for attending our workshop",
        "body": "Dear participants, thank you for joining our hands-on workshop on transformers and neural classifiers. The recorded video and notebook links are now live.",
        "label": "Primary",
        "source": "academic_workshop"
    },
    {
        "subject": "Apartment maintenance request status: Completed",
        "body": "Dear resident, the plumbing repair request #8201 for unit 4B was marked complete today by the building superintendent. Please let us know if issues persist.",
        "label": "Primary",
        "source": "administrative_housing"
    },
    {
        "subject": "Coffee catch-up next week?",
        "body": "Hey Maya, I saw your update about finishing the NLP project! Congrats! Would love to hear all about it over coffee next Tuesday if you're around campus.",
        "label": "Primary",
        "source": "personal_correspondence"
    },

    # =========================================================================
    # PROMOTIONS SAMPLES (Sales, discounts, newsletters, marketing, offers)
    # =========================================================================
    {
        "subject": "Flash Sale: 50% off all electronics ends tonight!",
        "body": "Do not miss out on our biggest weekend clearance event. Save up to 50% on headphones, smartwatches, and laptops. Use code FLASH50 at checkout.",
        "label": "Promotions",
        "source": "curated_ecommerce"
    },
    {
        "subject": "Exclusive Member Offer: Free shipping on your next order",
        "body": "As a valued rewards member, enjoy complimentary priority shipping on all orders over $25. Browse our autumn collection and treat yourself today.",
        "label": "Promotions",
        "source": "curated_ecommerce"
    },
    {
        "subject": "Your weekly digest: Top tech stories and tutorials",
        "body": "Here are the top trending articles in software engineering this week. Learn how modern compilers optimize vector operations. Click here to read full stories.",
        "label": "Promotions",
        "source": "curated_newsletter"
    },
    {
        "subject": "Upgrade to Pro: 30% discount on annual subscriptions",
        "body": "Unlock unlimited cloud storage, advanced analytics, and priority customer support. Upgrade your account today and lock in your discount for life.",
        "label": "Promotions",
        "source": "curated_saas"
    },
    {
        "subject": "New arrivals just dropped in our store!",
        "body": "Discover the latest athletic wear and running shoes designed for ultimate comfort and peak performance. Shop the new season drop now.",
        "label": "Promotions",
        "source": "curated_retail"
    },
    {
        "subject": "Black Friday sneak peek: VIP early access starts now",
        "body": "Beat the holiday rush! VIP members get early access to doorbuster prices across all home appliances and gadget bundles. Tap to shop before stock runs out.",
        "label": "Promotions",
        "source": "curated_ecommerce"
    },
    {
        "subject": "Special 2-for-1 pizza coupon inside!",
        "body": "Hungry? Order any large pizza tonight and get a second one completely free. Enter promo code BOGOPIZZA at checkout. Valid for delivery and takeout.",
        "label": "Promotions",
        "source": "curated_food_marketing"
    },
    {
        "subject": "Learn Cloud Computing with 70% off our bestseller course",
        "body": "Accelerate your career with hands-on labs and certification prep. Enroll this week to receive lifetime access at an unbeatable promotional price.",
        "label": "Promotions",
        "source": "curated_edtech"
    },
    {
        "subject": "Flights on sale: Round trips from $99",
        "body": "Plan your next weekend getaway with deeply discounted domestic fares. Book by midnight Wednesday to secure low seasonal rates.",
        "label": "Promotions",
        "source": "curated_travel"
    },
    {
        "subject": "Your monthly streaming recommendations",
        "body": "Check out the newest blockbusters and binge-worthy original series streaming this month. Update your watchlist and start streaming now.",
        "label": "Promotions",
        "source": "curated_streaming"
    },
    {
        "subject": "Final hours: Clearance sale up to 75% off",
        "body": "Last chance to save big on hundreds of clearance items before inventory refresh. Extra 15% off applied automatically in cart.",
        "label": "Promotions",
        "source": "curated_retail"
    },
    {
        "subject": "Earn double reward points on every purchase this weekend",
        "body": "Swipe your loyalty card at any partner retailer Saturday and Sunday to rack up 2x points toward cash vouchers and free merchandise.",
        "label": "Promotions",
        "source": "curated_loyalty"
    },
    {
        "subject": "Don't miss our Autumn Book Fair: Buy 2 Get 1 Free",
        "body": "Expand your personal library with thousands of fiction, science, and history titles. Enjoy buy 2 get 1 free across our entire catalog this week only.",
        "label": "Promotions",
        "source": "curated_retail"
    },
    {
        "subject": "Save $200 on modern standing desks and ergonomic chairs",
        "body": "Transform your home office workspace with ergonomic comfort. Use discount voucher WORKWELL at checkout for instant savings and free doorstep delivery.",
        "label": "Promotions",
        "source": "curated_ecommerce"
    },
    {
        "subject": "Special invitation: Join our exclusive beta tester community",
        "body": "Be the first to try cutting-edge AI productivity tools. Beta testers get complimentary 6 months premium access and direct input into feature roadmaps.",
        "label": "Promotions",
        "source": "curated_saas"
    },
    {
        "subject": "Weekend Hotel Deals: 40% off luxury beach resorts",
        "body": "Escape the routine with discounted seaside getaways. Book your sunny weekend retreat today and enjoy complimentary breakfast and spa credits.",
        "label": "Promotions",
        "source": "curated_travel"
    },
    {
        "subject": "Your favorite sneakers are back in stock!",
        "body": "The classic retro court sneakers you viewed are available again in all sizes. Grab your pair before inventory sells out.",
        "label": "Promotions",
        "source": "curated_retail"
    },
    {
        "subject": "Top 10 podcasts to boost your productivity this month",
        "body": "Discover inspiring conversations on leadership, deep focus, and creative habits. Listen to our curated episode playlist on Spotify or Apple Podcasts.",
        "label": "Promotions",
        "source": "curated_newsletter"
    },
    {
        "subject": "Get $15 off your first grocery delivery order",
        "body": "Fresh produce, dairy, and pantry essentials delivered straight to your kitchen in under 2 hours. Use code FRESH15 on orders above $35.",
        "label": "Promotions",
        "source": "curated_food_marketing"
    },
    {
        "subject": "Annual Cyber Security summit: Register with early-bird pricing",
        "body": "Connect with industry leaders in threat intelligence and cloud defense. Early-bird registration closes this Friday. Reserve your conference pass now.",
        "label": "Promotions",
        "source": "curated_events"
    },
    {
        "subject": "Exclusive presale tickets: Rock Arena Tour 2026",
        "body": "As a loyal music fan club member, unlock exclusive early ticket access starting at 9 AM tomorrow. Use your private promo pass: CONCERTFAN.",
        "label": "Promotions",
        "source": "curated_entertainment"
    },
    {
        "subject": "Upgrade your smartphone and get up to $400 trade-in credit",
        "body": "Trade in your current device and receive high valuation credits toward the latest 5G flagship phone. Includes 0% APR financing options.",
        "label": "Promotions",
        "source": "curated_telecom"
    },
    {
        "subject": "Save 25% on annual gym and wellness memberships",
        "body": "Kickstart your health goals with state-of-the-art fitness equipment, swimming pools, and group yoga sessions. Sign up online with code FITLIFE.",
        "label": "Promotions",
        "source": "curated_wellness"
    },
    {
        "subject": "Curated wine tasting box: 50% off introductory shipment",
        "body": "Taste award-winning vintages selected by master sommeliers. Get your first tasting flight at half price with free insulated temperature packaging.",
        "label": "Promotions",
        "source": "curated_lifestyle"
    },
    {
        "subject": "Winter coat clearance event: Up to 60% off outerwear",
        "body": "Stay warm without breaking the bank. Down jackets, trench coats, and thermal parkas on clearance while supplies last. Shop now.",
        "label": "Promotions",
        "source": "curated_retail"
    },
    {
        "subject": "Your monthly investing newsletter: Market trends and ETF analysis",
        "body": "Explore our analysts' breakdown of clean energy equities, semiconductor yields, and emerging market bond spreads. Read full insights online.",
        "label": "Promotions",
        "source": "curated_financial_newsletter"
    },
    {
        "subject": "Zero delivery fees on all restaurant takeout this week",
        "body": "Craving sushi, burgers, or pad thai? Enjoy unlimited free food delivery from hundreds of local dining spots with your DashPass subscription.",
        "label": "Promotions",
        "source": "curated_food_marketing"
    },
    {
        "subject": "Last chance: Claim your $10 birthday gift certificate",
        "body": "We noticed you haven't used your annual birthday shopping reward yet! Spend it on any purchase before it expires at the end of the month.",
        "label": "Promotions",
        "source": "curated_loyalty"
    },
    {
        "subject": "Master Deep Learning with Python: 85% discount course bundle",
        "body": "Complete 4 hands-on projects in CNNs, RNNs, and Transformer models. Certificate included. Claim your discount before midnight Sunday.",
        "label": "Promotions",
        "source": "curated_edtech"
    },
    {
        "subject": "Explore our new home decor and kitchen collection",
        "body": "Refresh your living space with minimalist ceramic vases, linen cushions, and artisan cookware. View the lookbook and enjoy free returns.",
        "label": "Promotions",
        "source": "curated_retail"
    },

    # =========================================================================
    # SPAM SAMPLES (Phishing, scam, financial fraud, lottery, urgency lures)
    # =========================================================================
    {
        "subject": "CONGRATULATIONS! You have won $1,000,000 in the International Lottery",
        "body": "Your email was selected as the lucky winner of the global promotional lottery jackpot. Reply immediately with your full name and bank routing number to claim your funds.",
        "label": "Spam",
        "source": "curated_lottery_scam"
    },
    {
        "subject": "URGENT: Your account has been suspended due to suspicious activity",
        "body": "We detected unauthorized login attempts on your banking profile. Click here immediately to verify your identity and prevent permanent account termination: http://secure-bank-login-update.com",
        "label": "Spam",
        "source": "curated_phishing"
    },
    {
        "subject": "Confidential Business Partnership Proposal: Funds Transfer of $14.5M",
        "body": "I am barrister Williams from the federal audit bureau. A late client left an unclaimed sum of fourteen million dollars. I need an honest overseas partner to receive these funds for a 40% share.",
        "label": "Spam",
        "source": "curated_advance_fee_scam"
    },
    {
        "subject": "You have an unclaimed package pending delivery #USPS-92810",
        "body": "Your parcel could not be delivered due to missing street number. Please pay the $1.95 redelivery fee by clicking here within 24 hours: http://track-package-usps-verify.info",
        "label": "Spam",
        "source": "curated_delivery_phishing"
    },
    {
        "subject": "Act immediately: Bitcoin payment required to unlock your files",
        "body": "All files on your computer have been encrypted with military-grade cipher. You must send 0.05 BTC to wallet 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa within 48 hours or all personal data will be deleted.",
        "label": "Spam",
        "source": "curated_ransom_extortion"
    },
    {
        "subject": "Make $5000 weekly working from home with no experience",
        "body": "Guaranteed income opportunity! Companies need data entry clerks to process simple forms online. No fees, no risk. Click http://make-fast-cash-daily.biz to register right now.",
        "label": "Spam",
        "source": "curated_job_scam"
    },
    {
        "subject": "Final Warning: IRS Tax Audit penalty notice #849102",
        "body": "This is an official communication from federal tax enforcement. An overdue balance of $3,450 has resulted in a warrant for arrest. Contact this number immediately to settle via prepaid cards.",
        "label": "Spam",
        "source": "curated_impersonation"
    },
    {
        "subject": "Hot singles in your area want to chat tonight!",
        "body": "Meet attractive women nearby looking for casual hookups and dating. Click here now to view private webcam photos and chat anonymously: http://meet-now-direct.xyz",
        "label": "Spam",
        "source": "curated_adult_spam"
    },
    {
        "subject": "You have been chosen for an exclusive survey ($500 reward)",
        "body": "Complete our 2-minute shopper survey and claim a complimentary $500 Amazon gift certificate instantly. Only 3 slots remaining! Click http://survey-rewards-winner.top",
        "label": "Spam",
        "source": "curated_giftcard_scam"
    },
    {
        "subject": "Immediate action required: Password expiration notice",
        "body": "Your company email password will expire in 2 hours. To maintain access to your corporate inbox and calendar, click http://portal-auth-verification.net to keep your current password.",
        "label": "Spam",
        "source": "curated_corporate_phishing"
    },
    {
        "subject": "Cheapest generic medications online with no prescription needed",
        "body": "Order genuine Viagra, Cialis, and pain relief pills at 80% discount. Dispatched discreetly in plain packaging worldwide. Visit http://rx-online-direct-pharmacy.cc",
        "label": "Spam",
        "source": "curated_pharmacy_spam"
    },
    {
        "subject": "Exclusive inheritance notification from the Estate of late Dr. Robert",
        "body": "I am seeking your consent to present you as next of kin to receive a deposit of USD 8,500,000 left dormant in a private vault. Reply with your private telephone number.",
        "label": "Spam",
        "source": "curated_advance_fee_scam"
    },
    {
        "subject": "Security Alert: Unauthorized login attempt from Russia",
        "body": "Someone accessed your Apple ID from IP 185.220.101.5 in Moscow. If this was not you, confirm your credentials immediately to lock your account: http://apple-id-verify-secure.pw",
        "label": "Spam",
        "source": "curated_phishing"
    },
    {
        "subject": "Invoice Attached: Outstanding payment of $1,840.50 due",
        "body": "Please find attached the past due invoice #INV-9401. Failure to remit payment within 24 hours will result in collection proceedings. Download attachment: invoice_macro.docm",
        "label": "Spam",
        "source": "curated_malware_lure"
    },
    {
        "subject": "Urgent: Direct deposit payroll information update needed",
        "body": "Employee HR Notification: Our payroll system encountered an error routing your upcoming salary. Enter your banking information at http://hr-portal-direct-deposit.info to avoid payment delays.",
        "label": "Spam",
        "source": "curated_corporate_phishing"
    },
    {
        "subject": "CLAIM YOUR FREE $1,000 WALMART SHOPPING CARD",
        "body": "You were chosen as our weekly shopper loyalty finalist! Click http://claim-walmart-giftcard.net and enter your cell phone number to receive your digital gift card PIN instantly.",
        "label": "Spam",
        "source": "curated_giftcard_scam"
    },
    {
        "subject": "I recorded you on your webcam watching explicit websites",
        "body": "I installed Pegasus malware on your device. I have videos of you split-screen with the sites you visited. Pay $1,200 in Monero within 48 hours or I send the video to all your email contacts.",
        "label": "Spam",
        "source": "curated_blackmail_extortion"
    },
    {
        "subject": "Urgent wire transfer request - Wire instructions enclosed",
        "body": "Are you at your desk? I need an urgent wire transfer executed for an acquisition closing today. Treat this as strictly confidential. Reply with confirmation once processed.",
        "label": "Spam",
        "source": "curated_ceo_fraud"
    },
    {
        "subject": "DHL Express: Your shipment #9481023 on hold at customs",
        "body": "Your international package cannot clear customs due to unpaid duty taxes of $3.50. Click http://dhl-customs-release.link to pay now and schedule immediate delivery.",
        "label": "Spam",
        "source": "curated_delivery_phishing"
    },
    {
        "subject": "Exclusive cryptocurrency investment club: 400% guaranteed return",
        "body": "Automated arbitrage AI algorithm generates passive profits every 60 seconds. Deposit 0.1 ETH and watch your balance grow. Join 10,000+ happy crypto millionaires now: http://crypto-profit-engine.io",
        "label": "Spam",
        "source": "curated_crypto_scam"
    },
    {
        "subject": "PayPal: Suspicious charge of $649.99 to Coinbase Inc",
        "body": "We flagged an unauthorized transaction on your PayPal wallet. If you did not authorize this charge, call fraud support immediately at +1-888-000-SCAM to reverse the charge.",
        "label": "Spam",
        "source": "curated_refund_phishing"
    },
    {
        "subject": "Grant notification: You were awarded $250,000 from UN Foundation",
        "body": "The United Nations poverty alleviation grant program has approved your email address for financial empowerment disbursement. Contact our disbursement agent via WhatsApp at +234...",
        "label": "Spam",
        "source": "curated_grant_fraud"
    },
    {
        "subject": "Notice of Impending Legal Action: Case Ref #CR-88192",
        "body": "A civil complaint has been registered against you in magistrate court. Click here to download legal summons and avoid automatic summary default judgment: http://court-summons-download.cc",
        "label": "Spam",
        "source": "curated_phishing"
    },
    {
        "subject": "Get rich quick with automated forex trading signals",
        "body": "Quit your 9-to-5 job today. Our secret Wall Street bot guarantees $2,000 daily profits on autopilot with 99.4% win rate. Sign up risk-free today: http://auto-forex-profits.top",
        "label": "Spam",
        "source": "curated_financial_scam"
    },
    {
        "subject": "Your Netflix membership has been put on hold",
        "body": "We were unable to validate your billing information for the next billing cycle. Update your credit card number at http://netflix-account-restart.com to continue watching your favorite movies.",
        "label": "Spam",
        "source": "curated_phishing"
    },
    {
        "subject": "Earn money by becoming a Mystery Shopper in your neighborhood",
        "body": "Receive a cashier's check for $2,500. Deposit it, keep $400 commission, and test Western Union transfer speed with the remainder. Reply with your mailing address.",
        "label": "Spam",
        "source": "curated_fake_check_scam"
    },
    {
        "subject": "Geek Squad Protection Plan Auto-Renewal Receipt: $499.00",
        "body": "Thank you for renewing your Total Tech Support plan for 3 years. $499.00 has been debited. To cancel and request a full refund, contact customer care at +1-800-FAKE-HELP.",
        "label": "Spam",
        "source": "curated_refund_phishing"
    },
    {
        "subject": "Metropolitan Police: Camera speeding violation citation #5810",
        "body": "Your vehicle exceeded speed limits by 24 mph in a school zone. View photographic evidence and pay the fine within 7 days to avoid license suspension: http://traffic-violations-portal.net",
        "label": "Spam",
        "source": "curated_impersonation"
    },
    {
        "subject": "Your Microsoft Office 365 cloud storage is almost full",
        "body": "You have exceeded 98% of your cloud mailbox quota. Incoming messages will be bounced. Click http://storage-expand-office365.online to increase your mailbox capacity for free.",
        "label": "Spam",
        "source": "curated_corporate_phishing"
    },
    {
        "subject": "Pre-approved for $50,000 unsecured personal loan: No credit check",
        "body": "Instant approval regardless of bad credit or bankruptcy. Low fixed interest rate. Wire an upfront processing fee of $250 to release your funds within 1 hour: http://quick-cash-loans-direct.biz",
        "label": "Spam",
        "source": "curated_loan_scam"
    },
]

def load_or_create_dataset(
    custom_csv_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw email data or initializes the documented curated starter dataset.
    If the companion Hugging Face dataset (data/dataset/train.csv) is available,
    it maps 'promotions' and 'spam' from it and blends with authentic 'Primary' emails.
    Performs data cleaning, deduplication, and stratified train/val/test splits.
    Saves splits to output_dir (defaults to data/processed).
    """
    dataset_dir = DATA_DIR / "dataset"
    ds_train_path = dataset_dir / "train.csv"
    ds_test_path = dataset_dir / "test.csv"

    if custom_csv_path and os.path.isfile(custom_csv_path):
        df = pd.read_csv(custom_csv_path)
        logger.info(f"Loaded custom raw data from {custom_csv_path} with {len(df)} records.")
    elif ds_train_path.is_file():
        logger.info(f"Found companion dataset at {dataset_dir}. Integrating companion dataset records...")
        ds_train_df = pd.read_csv(ds_train_path)
        
        # Extract promotions and spam from companion dataset
        promos = ds_train_df[ds_train_df["category"] == "promotions"][["subject", "body"]].copy()
        promos["label"] = "Promotions"
        promos["source"] = "hf_email_classification_dataset"

        spams = ds_train_df[ds_train_df["category"] == "spam"][["subject", "body"]].copy()
        spams["label"] = "Spam"
        spams["source"] = "hf_email_classification_dataset"

        # Extract authentic Primary emails from curated reference dataset
        primary_list = [d for d in STARTER_DATASET if d["label"] == "Primary"]
        primary_df = pd.DataFrame(primary_list)

        # Sample a balanced subset of promotions and spam to maintain good class balance
        sample_size = min(350, len(promos), len(spams))
        promos_sample = promos.sample(sample_size, random_state=random_state)
        spams_sample = spams.sample(sample_size, random_state=random_state)

        df = pd.concat([primary_df, promos_sample, spams_sample], ignore_index=True)
        logger.info(f"Assembled training corpus of {len(df)} records ({len(primary_df)} Primary, {len(promos_sample)} Promotions, {len(spams_sample)} Spam).")
    else:
        df = pd.DataFrame(STARTER_DATASET)
        logger.info(f"Initialized reference dataset with {len(df)} verified records.")

    # Validate required columns
    required_cols = {"subject", "body", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Standardize and filter labels
    df = df[df["label"].isin(TARGET_CATEGORIES)].copy()
    
    # Add combined text representation
    df["combined_text"] = df.apply(
        lambda r: combine_subject_and_body(r.get("subject", ""), r.get("body", "")),
        axis=1
    )
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    # Deduplication on cleaned text
    initial_count = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    dedup_count = len(df)
    logger.info(f"Deduplicated dataset from {initial_count} to {dedup_count} unique rows.")

    # Stratified Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"]
    )

    out = output_dir or (DATA_DIR / "processed")
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    logger.info(f"Saved splits to {out} (train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)})")

    return train_df, val_df, test_df

