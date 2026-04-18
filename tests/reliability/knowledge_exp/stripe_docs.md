# Stripe Payments Documentation

## Payout Schedule
Payouts run daily on business days by default, with a 2-day rolling reserve for new accounts. You can adjust the schedule to weekly or monthly in Dashboard → Balance → Payout settings.

## Failed Subscription Payments
Stripe automatically retries failed subscription charges using Smart Retries, which uses machine learning to pick the best retry time. Default policy is 4 retries over 3 weeks before marking the subscription past_due.

## Processing Fee Refunds
The payment processing fee is not returned when you refund a charge. This applies to both full and partial refunds. The fee for the original transaction is retained by Stripe.

## Responding to Chargebacks and Disputes
Go to Dashboard → Disputes, open the disputed charge, and upload evidence (receipts, customer communication, shipping proof) within 7 days. Stripe submits the evidence to the card network on your behalf.

## International Credit Cards
Stripe accepts Visa, Mastercard, American Express, Discover, and most region-specific cards. International card transactions incur an additional 1.5% fee on top of the standard processing rate.

## Setting Up Recurring Billing
Use Stripe Subscriptions. Create a Product in Dashboard → Products, add Pricing (monthly/yearly), then use the Subscriptions API or Checkout to subscribe customers. Webhooks notify your app of billing events.

## Test Mode API Keys
Test mode lets you integrate Stripe without real money moving. Use the test secret key (starts with sk_test_) in your development environment. Test cards like 4242 4242 4242 4242 simulate successful charges.

## Webhook Security
Every webhook request includes a Stripe-Signature header. Use your endpoint's signing secret (Dashboard → Webhooks) to verify the signature with Stripe's library: stripe.webhooks.constructEvent(payload, sig, secret).

## Sales Tax and VAT Collection
Stripe Tax is a paid add-on that calculates and collects the correct sales tax, VAT, or GST based on the customer's location. Enable it in Dashboard → Tax and add tax_behavior to your price objects.

## Charging in Local Currency
Set the currency field when creating a Payment Intent or Price. Stripe supports 135+ currencies. Settlement currency depends on your account — you can enable multi-currency settlement in Balance settings.

## Fraud Prevention with Radar
Stripe Radar uses machine learning on the Stripe network to score each payment for fraud risk. Charges above a risk threshold are blocked or sent for review. Radar rules can be customized in Dashboard → Radar.

## Account Verification
Stripe verifies business and identity information to comply with financial regulations. Upload a government-issued ID and business documents in Dashboard → Settings → Business. Verification typically completes within 1-2 business days.

## Customer Self-Service Portal
The Customer Portal is a pre-built page where customers can update payment methods, view invoices, and cancel subscriptions. Enable it in Dashboard → Settings → Billing and generate a portal link via the API.

## Automatic Invoice Emails
Invoices finalized in Stripe automatically email a PDF to the customer's email address if 'Email invoices to customers' is enabled under Dashboard → Settings → Invoicing. Disable per invoice with auto_advance=false.

## Marketplace Split Payments with Connect
Use Stripe Connect. Create connected accounts for each party, then pass application_fee_amount and transfer_data[destination] on the charge to split the payment. Standard, Express, and Custom account types differ in onboarding depth.

## Refund Time Window
Refunds can be issued up to 180 days after the original charge. After 180 days, refunds are no longer possible through Stripe — you'd need to send funds through a different channel.

## Transaction Fees and Pricing
US: 2.9% + 30¢ for online card payments. International cards add 1.5%. Currency conversion adds 1%. ACH is 0.8% capped at $5. Custom pricing available for volumes above $80K/month.

## Stripe Checkout vs Elements
Checkout is a hosted, pre-built payment page — fastest to integrate, PCI compliance handled. Elements gives full control over the form UI, embedded in your own site — requires more code but fits custom designs.

## Saving Cards for Future Charges
Attach a PaymentMethod to a Customer object via setup_future_usage or a SetupIntent. Future charges use the stored PaymentMethod without re-prompting the customer. Strong Customer Authentication may apply in some regions.

## Payout Holds
Payouts can be held due to unusually high risk scores on recent charges, pending disputes, incomplete account verification, or a temporary reserve Stripe has placed on your account. Check Dashboard → Balance for specifics.
