"""Run from project folder: python -m unittest discover -s tests -v"""

import asyncio
import json
import unittest

import httpx

from app import (
    CLOUD_API_URL,
    EMERGENCY,
    MANUAL,
    ROUTINE,
    TRIAGE_SCHEMA,
    InvalidTriageResponse,
    apply_safety_overlay,
    call_cloud,
    call_local,
    make_mock_transport,
    parse_and_validate,
    run_triage,
)

VALID = {
    "is_critical_emergency": False,
    "detected_symptoms": ["mild cough"],
    "clinical_reasoning_summary": "Only mild cough was reported.",
    "routing_destination": ROUTINE,
}


class SchemaTests(unittest.TestCase):
    def test_json_loads_and_exact_keys(self):
        self.assertEqual(parse_and_validate(json.dumps(VALID)), VALID)

    def test_malformed_json_rejected(self):
        with self.assertRaises(InvalidTriageResponse):
            parse_and_validate("```json\n{}\n```")

    def test_extra_key_rejected(self):
        with self.assertRaises(InvalidTriageResponse):
            parse_and_validate(json.dumps({**VALID, "extra": "not allowed"}))

    def test_missing_key_rejected(self):
        invalid = VALID.copy()
        del invalid["detected_symptoms"]
        with self.assertRaises(InvalidTriageResponse):
            parse_and_validate(json.dumps(invalid))

    def test_non_boolean_rejected(self):
        with self.assertRaises(InvalidTriageResponse):
            parse_and_validate(json.dumps({**VALID, "is_critical_emergency": "false"}))

    def test_emergency_route_always_consistent(self):
        inconsistent = {**VALID, "is_critical_emergency": True}
        result = apply_safety_overlay(
            inconsistent, "An example without warning phrases."
        )
        self.assertEqual(result["routing_destination"], EMERGENCY)

    def test_reported_red_flag_overrides_under_triage(self):
        result = apply_safety_overlay(VALID, "I have crushing chest pain.")
        self.assertTrue(result["is_critical_emergency"])
        self.assertEqual(result["routing_destination"], EMERGENCY)

    def test_simple_negation(self):
        self.assertEqual(
            apply_safety_overlay(VALID, "I have no severe chest pain.")[
                "routing_destination"
            ],
            ROUTINE,
        )


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cloud_json_mode_and_exact_model(self):
        captured = {}

        def mocked(request):
            self.assertEqual(str(request.url), CLOUD_API_URL)
            captured.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(VALID)}}]}
            )

        response = await call_cloud(
            "I have a mild cough", "dummy", transport=httpx.MockTransport(mocked)
        )
        self.assertEqual(parse_and_validate(response), VALID)
        self.assertEqual(captured["model"], "gpt-4o-mini")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][1]["role"], "user")

    async def test_ollama_native_schema(self):
        captured = {}

        def mocked(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": json.dumps(VALID)}})

        response = await call_local(
            "I have a mild cough", transport=httpx.MockTransport(mocked)
        )
        self.assertEqual(parse_and_validate(response), VALID)
        self.assertEqual(captured["format"], TRIAGE_SCHEMA)
        self.assertFalse(captured["stream"])

    async def test_cloud_to_local_fallback_503(self):
        result = await run_triage(
            "I have crushing chest pain and cannot breathe.",
            api_key="fake-key",
            cloud_transport=make_mock_transport("cloud", unavailable=True),
            local_transport=make_mock_transport("local"),
        )
        self.assertEqual(result.source, "ollama_fallback")
        self.assertEqual(result.data["routing_destination"], EMERGENCY)
        self.assertEqual(len(result.data), 4)  # Four required fields in actual brief
        self.assertTrue(result.errors)

    async def test_cloud_timeout_falls_back(self):
        async def slow(request):
            raise httpx.ReadTimeout("simulated timeout")

        result = await run_triage(
            "I have a mild cough",
            api_key="fake-key",
            cloud_transport=httpx.MockTransport(slow),
            local_transport=make_mock_transport("local"),
        )
        self.assertEqual(result.source, "ollama_fallback")
        self.assertEqual(result.data["routing_destination"], ROUTINE)

    async def test_invalid_cloud_json_falls_back(self):
        def badly_formatted(request):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "not json"}}]}
            )

        result = await run_triage(
            "I have a mild cough",
            api_key="fake-key",
            cloud_transport=httpx.MockTransport(badly_formatted),
            local_transport=make_mock_transport("local"),
        )
        self.assertEqual(result.source, "ollama_fallback")

    async def test_both_unavailable_fails_closed(self):
        def no_service(request):
            return httpx.Response(503, json={"error": "service down"})

        result = await run_triage(
            "I have a mild cough",
            api_key="fake-key",
            cloud_transport=httpx.MockTransport(no_service),
            local_transport=httpx.MockTransport(no_service),
        )
        self.assertEqual(result.source, "manual_failsafe")
        self.assertEqual(result.data["routing_destination"], MANUAL)
        self.assertIn("unknown", result.data["clinical_reasoning_summary"].lower())

    async def test_cloud_success_latency_comparison(self):
        result = await run_triage(
            "I have a mild cough",
            api_key="fake-key",
            cloud_transport=make_mock_transport("cloud"),
            local_transport=make_mock_transport("local"),
            compare=True,
        )
        self.assertEqual(result.source, "cloud")
        self.assertIn("cloud", result.timings_ms)
        self.assertIn("local", result.timings_ms)


if __name__ == "__main__":
    unittest.main()
