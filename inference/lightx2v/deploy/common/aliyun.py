import asyncio
import json
import os
import sys

from alibabacloud_dypnsapi20170525 import models as dypnsapi_models
from alibabacloud_dypnsapi20170525.client import Client
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_util import models as util_models
from loguru import logger


class AlibabaCloudClient:
    def __init__(self):
        config = openapi_models.Config(
            access_key_id=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
            access_key_secret=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
            https_proxy=os.getenv("auth_https_proxy", None),
        )
        self.client = Client(config)
        self.runtime = util_models.RuntimeOptions()

    def check_ok(self, res, prefix):
        if not isinstance(res, dict):
            logger.warning("{} returned an invalid response type", prefix)
            return False

        status_code = res.get("statusCode")
        body = res.get("body")
        if status_code != 200:
            logger.warning("{} failed with HTTP status {}", prefix, status_code)
            return False
        if not isinstance(body, dict) or "Code" not in body or "Success" not in body:
            logger.warning("{} returned a malformed response body", prefix)
            return False
        if body["Code"] != "OK" or body["Success"] is not True:
            logger.warning("{} failed with provider code {}", prefix, body.get("Code"))
            return False

        logger.info("{} succeeded", prefix)
        return True

    async def send_sms(self, phone_number):
        try:
            req = dypnsapi_models.SendSmsVerifyCodeRequest(
                phone_number=phone_number,
                sign_name="速通互联验证服务",
                template_code="100001",
                template_param=json.dumps({"code": "##code##", "min": "5"}),
                valid_time=300,
            )
            res = await self.client.send_sms_verify_code_with_options_async(req, self.runtime)
            ok = self.check_ok(res.to_map(), "AlibabaCloudClient send sms")
            logger.info("AlibabaCloudClient send SMS completed: {}", ok)
            return ok

        except Exception as e:
            logger.warning("AlibabaCloudClient send SMS failed: {}", type(e).__name__)
            return False

    async def check_sms(self, phone_number, verify_code):
        try:
            req = dypnsapi_models.CheckSmsVerifyCodeRequest(
                phone_number=phone_number,
                verify_code=verify_code,
            )
            res = await self.client.check_sms_verify_code_with_options_async(req, self.runtime)
            ok = self.check_ok(res.to_map(), "AlibabaCloudClient check sms")
            logger.info("AlibabaCloudClient check SMS completed: {}", ok)
            return ok

        except Exception as e:
            logger.warning("AlibabaCloudClient check SMS failed: {}", type(e).__name__)
            return False


async def test(args):
    assert len(args) in [1, 2], "Usage: python aliyun_sms.py <phone_number> [verify_code]"
    phone_number = args[0]
    client = AlibabaCloudClient()
    if len(args) == 1:
        await client.send_sms(phone_number)
    else:
        await client.check_sms(phone_number, args[1])


if __name__ == "__main__":
    asyncio.run(test(sys.argv[1:]))
