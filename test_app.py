
import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(
            data["service"],
            "student-result-prediction"
        )

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85,
                "assignment_marks": 88,
                "previous_score": 80
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["prediction"], "PASS")
        self.assertEqual(data["prediction_code"], 1)

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 55,
                "internal_marks": 30,
                "assignment_marks": 40,
                "previous_score": 35
            }
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["prediction"], "FAIL")
        self.assertEqual(data["prediction_code"], 0)

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85
            }
        )

        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertIn("missing_fields", data)


if __name__ == "__main__":
    unittest.main()
