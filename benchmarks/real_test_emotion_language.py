"""Real tests for emotion-detection and language-detect packs.

Self-seed accuracy is trivially circular. The honest questions:

LANGUAGE-DETECT — does it actually route foreign text to the right language?
  Feed 20 real non-English samples (es, fr, ja, ar) and check routing.

EMOTION-DETECTION — can it disambiguate close emotions from overlapping vocab?
  Feed hand-crafted unambiguous emotion queries and check routing.
"""
import json, shutil
from pathlib import Path
import microresolve

THRESHOLD = 1.5
ROOT = Path("/tmp/real_test")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)

# ───────────────────────────────────────────────────────────────────────
# LANGUAGE-DETECT — feed actual non-English text
# ───────────────────────────────────────────────────────────────────────
print("=" * 72)
print("LANGUAGE-DETECT — real test on non-English text")
print("=" * 72)

shutil.copytree("packs/language-detect", ROOT / "language-detect" / "language-detect")
ns = microresolve.MicroResolve(data_dir=str(ROOT / "language-detect")).namespace("language-detect")

# 20 samples each from real-world multilingual text
LANG_PROBES = {
    "spanish": [
        "buenos días, ¿cómo está usted hoy?",
        "me gustaría reservar una mesa para dos personas",
        "el clima está muy bueno esta tarde",
        "no entiendo lo que dijiste",
        "pueden enviar la factura por correo electrónico",
        "quiero cancelar mi suscripción",
        "tengo una pregunta sobre el pedido",
        "gracias por su ayuda",
    ],
    "french": [
        "bonjour, comment allez-vous aujourd'hui",
        "je voudrais réserver une table pour deux",
        "le temps est très beau cet après-midi",
        "je ne comprends pas ce que vous dites",
        "pouvez-vous envoyer la facture par email",
        "je veux annuler mon abonnement",
        "j'ai une question concernant ma commande",
        "merci beaucoup pour votre aide",
    ],
    "german": [
        "guten tag, wie geht es ihnen heute",
        "ich möchte einen tisch für zwei reservieren",
        "das wetter ist heute sehr schön",
        "ich verstehe nicht was sie sagen",
        "können sie die rechnung per email schicken",
        "ich möchte mein abonnement kündigen",
        "ich habe eine frage zu meiner bestellung",
        "vielen dank für ihre hilfe",
    ],
    "japanese": [
        "こんにちは、お元気ですか",
        "二名でテーブルを予約したいです",
        "今日の天気は素晴らしいです",
        "あなたの言っていることがわかりません",
        "領収書をメールで送ってもらえますか",
        "サブスクリプションをキャンセルしたいです",
        "注文について質問があります",
        "ご協力ありがとうございます",
    ],
}

correct = 0
total = 0
errors = []
for true_lang, samples in LANG_PROBES.items():
    expected = f"detect_{true_lang}"
    pack_hit = 0
    for q in samples:
        r = ns.resolve(q)
        top = next((i for i in r.intents if i.score >= THRESHOLD), None)
        top_id = top.id if top else "—"
        total += 1
        if top_id == expected:
            correct += 1
            pack_hit += 1
        else:
            errors.append((q, expected, top_id, top.score if top else 0))
    print(f"  {true_lang:10s}: {pack_hit}/{len(samples)} routed to {expected}")

print(f"\n  TOTAL: {correct}/{total} = {correct/total:.1%}")
if errors[:5]:
    print(f"\n  First 5 mis-routes:")
    for q, exp, got, sc in errors[:5]:
        print(f"    '{q[:50]}' → expected {exp}, got {got} ({sc:.2f})")

# ───────────────────────────────────────────────────────────────────────
# EMOTION-DETECTION — adversarial in-domain
# ───────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("EMOTION-DETECTION — disambiguation test on unambiguous queries")
print("=" * 72)

shutil.copytree("packs/emotion-detection", ROOT / "emotion-detection" / "emotion-detection")
ns2 = microresolve.MicroResolve(data_dir=str(ROOT / "emotion-detection")).namespace("emotion-detection")

EMOTION_PROBES = [
    # clearly anxious
    ("i'm really worried this won't work out before the deadline", "anxious_worried"),
    ("i'm scared something bad might happen", "anxious_worried"),
    ("i can't stop worrying about the surgery tomorrow", "anxious_worried"),
    # clearly frustrated / angry
    ("this is the third time the app crashed, i'm so angry", "frustrated_angry"),
    ("absolute joke of a service, fix your bugs", "frustrated_angry"),
    ("furious that my package still hasn't arrived", "frustrated_angry"),
    # confused
    ("i have no idea how to set up this thing", "confused_lost"),
    ("the instructions don't make any sense to me", "confused_lost"),
    ("which button should i click i'm totally lost", "confused_lost"),
    # disappointed
    ("expected so much better from this product", "disappointed_let_down"),
    ("really let down by the customer service today", "disappointed_let_down"),
    ("thought this would be great but i was wrong", "disappointed_let_down"),
    # distressed / urgent
    ("emergency, i need help right now please", "distressed_urgent"),
    ("urgent — my account has been hacked", "distressed_urgent"),
    # satisfied
    ("absolutely love this, exactly what i wanted", "satisfied_positive"),
    ("five stars, very happy with the experience", "satisfied_positive"),
    ("perfect product, exactly as described", "satisfied_positive"),
    # neutral
    ("what time does the store open", "neutral_informational"),
    ("which version of the software do i need", "neutral_informational"),
    ("how do i reset my password", "neutral_informational"),
]

e_correct = 0
e_top3 = 0
e_errors = []
for q, expected in EMOTION_PROBES:
    r = ns2.resolve(q)
    top = next((i for i in r.intents if i.score >= THRESHOLD), None)
    top_id = top.id if top else "—"
    top3_ids = [i.id for i in r.intents[:3]]
    if top_id == expected:
        e_correct += 1
    if expected in top3_ids:
        e_top3 += 1
    if top_id != expected:
        e_errors.append((q, expected, top_id, top.score if top else 0))

print(f"  top-1: {e_correct}/{len(EMOTION_PROBES)} = {e_correct/len(EMOTION_PROBES):.1%}")
print(f"  top-3: {e_top3}/{len(EMOTION_PROBES)} = {e_top3/len(EMOTION_PROBES):.1%}")
if e_errors:
    print(f"\n  Mis-routes:")
    for q, exp, got, sc in e_errors:
        print(f"    '{q[:55]}'  expected={exp}  got={got} ({sc:.2f})")
