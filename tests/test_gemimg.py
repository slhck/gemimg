import json
import unittest

import httpx

from gemimg import GemImg

PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY"
    "42YAAAAASUVORK5CYII="
)


def make_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "responseId": "response-id",
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 8,
            },
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "thought": True,
                                "inlineData": {"data": PNG_1X1},
                            },
                            {"inlineData": {"data": PNG_1X1}},
                        ]
                    },
                }
            ],
        },
        request=request,
    )


class GemImgRequestTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def handler(request):
            self.requests.append(request)
            return make_response(request)

        self.client = httpx.Client(transport=httpx.MockTransport(handler))

    def tearDown(self):
        self.client.close()

    def test_flash_31_uses_stable_endpoint_and_new_image_options(self):
        generator = GemImg(api_key="test", client=self.client)

        result = generator.generate(
            "test",
            aspect_ratio="1:8",
            image_size="512",
            thinking_level="high",
            save=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(
            str(self.requests[0].url),
            "https://generativelanguage.googleapis.com/v1/models/"
            "gemini-3.1-flash-image:generateContent",
        )
        payload = json.loads(self.requests[0].content)
        config = payload["generationConfig"]
        self.assertEqual(config["responseModalities"], ["IMAGE"])
        self.assertEqual(
            config["responseFormat"]["image"],
            {"aspectRatio": "1:8", "imageSize": "512"},
        )
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "high"})

    def test_legacy_flash_omits_fixed_image_size(self):
        generator = GemImg(
            api_key="test",
            client=self.client,
            model="gemini-2.5-flash-image",
        )

        generator.generate("test", save=False)

        payload = json.loads(self.requests[0].content)
        image_config = payload["generationConfig"]["responseFormat"]["image"]
        self.assertNotIn("imageSize", image_config)

    def test_model_specific_options_are_validated(self):
        pro = GemImg(api_key="test", client=self.client, model="gemini-3-pro-image")
        lite = GemImg(
            api_key="test",
            client=self.client,
            model="gemini-3.1-flash-lite-image",
        )

        with self.assertRaisesRegex(ValueError, "invalid"):
            pro.generate("test", aspect_ratio="1:8", save=False)
        with self.assertRaisesRegex(ValueError, "image_size"):
            lite.generate("test", image_size="2K", save=False)


if __name__ == "__main__":
    unittest.main()
