#!/usr/bin/env python3
"""
Script de test rapide pour l'API Churn Prediction.

Usage:
    python test_api.py
"""

import requests
import json
from time import sleep

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "demo-key-123"

def test_health():
    """Test health check"""
    print("\n" + "="*70)
    print("  TEST 1: Health Check")
    print("="*70)

    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"✅ API Version: {data['api_version']}")
            print(f"✅ Model Version: {data['model_version']}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        print("Assurez-vous que l'API est lancée: docker-compose up -d")
        return False


def test_metrics():
    """Test métriques du modèle"""
    print("\n" + "="*70)
    print("  TEST 2: Métriques du Modèle")
    print("="*70)

    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ F1-Score: {data['metrics']['test_f1']:.4f}")
            print(f"✅ Precision: {data['metrics']['test_precision']:.4f}")
            print(f"✅ Recall: {data['metrics']['test_recall']:.4f}")
            print(f"✅ ROC-AUC: {data['metrics']['test_roc_auc']:.4f}")
            print(f"✅ Threshold: {data['threshold']:.3f}")
            print(f"✅ Modèles: {', '.join(data['models'])}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_prediction():
    """Test prédiction simple"""
    print("\n" + "="*70)
    print("  TEST 3: Prédiction Simple")
    print("="*70)

    # Charger les données de test
    with open('test_customer.json', 'r') as f:
        customer = json.load(f)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Customer ID: {result.get('customerID', 'N/A')}")
            print(f"✅ Churn Probability: {result['churn_probability']:.2%}")
            print(f"✅ Churn Prediction: {'OUI' if result['churn_prediction'] == 1 else 'NON'}")
            print(f"✅ Risk Level: {result['risk_level']}")
            print(f"✅ Confidence: {result['confidence']:.2%}")
            print(f"\n📋 Action Recommandée:")
            print(f"   {result['recommended_action']}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_without_api_key():
    """Test sécurité (sans API key)"""
    print("\n" + "="*70)
    print("  TEST 4: Sécurité (Sans API Key)")
    print("="*70)

    with open('test_customer.json', 'r') as f:
        customer = json.load(f)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer
            # Pas de header X-API-Key
        )

        if response.status_code == 401:
            print("✅ Sécurité OK: Requête bloquée sans API key")
            return True
        else:
            print(f"❌ Sécurité FAIL: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_latency():
    """Test latence"""
    print("\n" + "="*70)
    print("  TEST 5: Performance (Latence)")
    print("="*70)

    with open('test_customer.json', 'r') as f:
        customer = json.load(f)

    latencies = []
    n_requests = 10

    print(f"Exécution de {n_requests} requêtes...")

    for i in range(n_requests):
        import time
        start = time.time()

        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            headers={"X-API-Key": API_KEY}
        )

        latency = (time.time() - start) * 1000  # en ms
        latencies.append(latency)
        print(f"  Request {i+1}: {latency:.0f}ms")

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    print(f"\n✅ Latence moyenne: {avg_latency:.0f}ms")
    print(f"✅ Latence P95: {p95_latency:.0f}ms")

    if avg_latency < 100:
        print("✅ Performance: EXCELLENTE (<100ms)")
    elif avg_latency < 200:
        print("⚠️  Performance: BONNE (100-200ms)")
    else:
        print("❌ Performance: À AMÉLIORER (>200ms)")

    return True


def main():
    """Exécuter tous les tests"""
    print("\n" + "="*70)
    print("  🧪 TESTS DE L'API CHURN PREDICTION")
    print("="*70)

    tests = [
        ("Health Check", test_health),
        ("Métriques", test_metrics),
        ("Prédiction", test_prediction),
        ("Sécurité", test_without_api_key),
        ("Performance", test_latency)
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Test '{name}' failed: {e}")
            results.append((name, False))

        sleep(0.5)  # Pause entre les tests

    # Résumé
    print("\n" + "="*70)
    print("  📊 RÉSUMÉ DES TESTS")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Score: {passed}/{total} tests réussis")

    if passed == total:
        print("\n  🎉 TOUS LES TESTS SONT PASSÉS!")
        print("\n  L'API est prête pour la production! 🚀")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) échoué(s)")
        print("\n  Vérifiez les logs: docker-compose logs -f churn-api")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
