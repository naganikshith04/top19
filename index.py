#!/usr/bin/env python3
"""
AWS Bedrock Claude Model Deployment and Inference Script
This script handles Claude model inference via AWS Bedrock with automatic resource management.
"""

import boto3
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BedrockClaudeManager:
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize the Bedrock Claude manager

        Args:
            region_name: AWS region for deployment (ensure Bedrock is available)
        """
        self.region_name = region_name
        self.bedrock = boto3.client("bedrock", region_name=region_name)
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)
        self.provisioned_throughput_arn = None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available Claude models in Bedrock

        Returns:
            List of available models
        """
        try:
            response = self.bedrock.list_foundation_models()
            claude_models = [
                model
                for model in response.get("modelSummaries", [])
                if "claude" in model.get("modelId", "").lower()
            ]

            logger.info(f"Found {len(claude_models)} Claude models available")
            for model in claude_models:
                logger.info(f"- {model.get('modelId')} ({model.get('modelName')})")

            return claude_models

        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise

    def create_provisioned_throughput(
        self, model_id: str, model_units: int = 1, commitment_duration: str = None
    ) -> str:
        """
        Create provisioned throughput for consistent performance (optional)
        Note: This incurs additional costs but provides guaranteed capacity

        Args:
            model_id: The Claude model ID
            model_units: Number of model units to provision
            commitment_duration: Optional commitment duration for cost savings

        Returns:
            Provisioned throughput ARN
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            throughput_name = f"claude-throughput-{timestamp}"

            create_params = {
                "provisionedModelName": throughput_name,
                "modelId": model_id,
                "modelUnits": model_units,
            }

            if commitment_duration:
                create_params["commitmentDuration"] = commitment_duration

            response = self.bedrock.create_provisioned_model_throughput(**create_params)

            throughput_arn = response["provisionedModelArn"]
            logger.info(f"Provisioned throughput creation initiated: {throughput_name}")
            logger.info(f"ARN: {throughput_arn}")

            # Wait for provisioned throughput to be ready
            self.wait_for_provisioned_throughput(throughput_arn)
            self.provisioned_throughput_arn = throughput_arn

            return throughput_arn

        except Exception as e:
            logger.error(f"Failed to create provisioned throughput: {str(e)}")
            raise

    def wait_for_provisioned_throughput(
        self, throughput_arn: str, max_wait_time: int = 1800
    ):
        """
        Wait for provisioned throughput to be ready

        Args:
            throughput_arn: ARN of the provisioned throughput
            max_wait_time: Maximum time to wait in seconds (default 30 minutes)
        """
        logger.info("Waiting for provisioned throughput to be ready...")
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                response = self.bedrock.get_provisioned_model_throughput(
                    provisionedModelId=throughput_arn
                )
                status = response["status"]

                if status == "InService":
                    logger.info("Provisioned throughput is now ready!")
                    return
                elif status == "Failed":
                    raise Exception(
                        f"Provisioned throughput failed: {response.get('failureMessage', 'Unknown error')}"
                    )
                elif status in ["Creating", "Updating"]:
                    logger.info(f"Provisioned throughput status: {status}. Waiting...")
                    time.sleep(60)  # Check every minute
                else:
                    logger.warning(f"Unexpected status: {status}")
                    time.sleep(60)

            except Exception as e:
                logger.error(f"Error checking provisioned throughput status: {str(e)}")
                raise

        raise Exception(
            f"Provisioned throughput did not become ready within {max_wait_time} seconds"
        )

    def run_inference(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_provisioned: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference with Claude model

        Args:
            model_id: The Claude model ID or provisioned throughput ARN
            messages: List of message objects for conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            use_provisioned: Whether to use provisioned throughput

        Returns:
            Inference results
        """
        try:
            # Use provisioned throughput ARN if available and requested
            if use_provisioned and self.provisioned_throughput_arn:
                model_identifier = self.provisioned_throughput_arn
                logger.info(f"Using provisioned throughput for inference")
            else:
                model_identifier = model_id
                logger.info(f"Using on-demand model: {model_id}")

            # Prepare the request body for Claude
            request_body = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31",
            }

            logger.info("Running inference...")
            logger.info(f"Input: {json.dumps(messages, indent=2)}")

            response = self.bedrock_runtime.invoke_model(
                modelId=model_identifier,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            logger.info("Inference completed successfully")
            logger.info(f"Response: {json.dumps(response_body, indent=2)}")

            return {
                "status": "success",
                "model_id": model_identifier,
                "response": response_body,
                "usage": response_body.get("usage", {}),
                "response_metadata": response.get("ResponseMetadata", {}),
            }

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return {"status": "error", "error": str(e), "model_id": model_id}

    def run_streaming_inference(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_provisioned: bool = False,
    ) -> Dict[str, Any]:
        """
        Run streaming inference with Claude model

        Args:
            model_id: The Claude model ID or provisioned throughput ARN
            messages: List of message objects for conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            use_provisioned: Whether to use provisioned throughput

        Returns:
            Inference results
        """
        try:
            # Use provisioned throughput ARN if available and requested
            if use_provisioned and self.provisioned_throughput_arn:
                model_identifier = self.provisioned_throughput_arn
            else:
                model_identifier = model_id

            request_body = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31",
            }

            logger.info("Starting streaming inference...")

            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=model_identifier,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            # Process streaming response
            full_response = ""
            for event in response["body"]:
                if "chunk" in event:
                    chunk = json.loads(event["chunk"]["bytes"].decode())
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {}).get("text", "")
                        full_response += delta
                        print(delta, end="", flush=True)  # Real-time output

            print()  # New line after streaming

            return {
                "status": "success",
                "model_id": model_identifier,
                "response": {"content": [{"text": full_response}]},
                "full_text": full_response,
            }

        except Exception as e:
            logger.error(f"Streaming inference failed: {str(e)}")
            return {"status": "error", "error": str(e), "model_id": model_id}

    def cleanup_resources(self):
        """
        Clean up provisioned throughput if created
        """
        if self.provisioned_throughput_arn:
            try:
                logger.info("Cleaning up provisioned throughput...")
                self.bedrock.delete_provisioned_model_throughput(
                    provisionedModelId=self.provisioned_throughput_arn
                )
                logger.info("Provisioned throughput deletion initiated")
            except Exception as e:
                logger.error(f"Failed to delete provisioned throughput: {str(e)}")

    def run_inference_workflow(
        self,
        model_configs: List[Dict[str, Any]],
        conversation_messages: List[Dict[str, Any]],
        use_provisioned: bool = False,
        streaming: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete workflow: optionally provision throughput, run inference, cleanup

        Args:
            model_configs: List of model configurations to test
            conversation_messages: Messages for the conversation
            use_provisioned: Whether to create provisioned throughput
            streaming: Whether to use streaming inference

        Returns:
            Results from all models
        """
        results = {}

        try:
            logger.info("Starting Claude inference workflow...")

            # List available models first
            self.list_available_models()

            for model_config in model_configs:
                model_id = model_config["model_id"]
                logger.info(f"\n{'='*50}")
                logger.info(f"Testing model: {model_id}")
                logger.info(f"{'='*50}")

                try:
                    # Optionally create provisioned throughput
                    if use_provisioned:
                        self.create_provisioned_throughput(
                            model_id=model_id,
                            model_units=model_config.get("model_units", 1),
                        )

                    # Run inference
                    if streaming:
                        inference_results = self.run_streaming_inference(
                            model_id=model_id,
                            messages=conversation_messages,
                            max_tokens=model_config.get("max_tokens", 1000),
                            temperature=model_config.get("temperature", 0.7),
                            use_provisioned=use_provisioned,
                        )
                    else:
                        inference_results = self.run_inference(
                            model_id=model_id,
                            messages=conversation_messages,
                            max_tokens=model_config.get("max_tokens", 1000),
                            temperature=model_config.get("temperature", 0.7),
                            use_provisioned=use_provisioned,
                        )

                    results[model_id] = inference_results

                except Exception as e:
                    logger.error(f"Failed to test model {model_id}: {str(e)}")
                    results[model_id] = {"status": "error", "error": str(e)}

                finally:
                    # Cleanup provisioned resources for this model
                    if use_provisioned:
                        self.cleanup_resources()
                        self.provisioned_throughput_arn = None

            return results

        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
        finally:
            # Final cleanup
            self.cleanup_resources()


def main():
    """
    Example usage with the specific Claude models requested
    """
    # Model configurations for the requested models
    model_configs = [
        {
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "max_tokens": 1000,
            "temperature": 0.7,
            "model_units": 1,  # For provisioned throughput if needed
        },
        {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "max_tokens": 1000,
            "temperature": 0.7,
            "model_units": 1,
        },
    ]

    # Sample conversation for testing
    conversation_messages = [
        {
            "role": "user",
            "content": "Hello! Can you explain what you are and provide a brief overview of your capabilities?",
        }
    ]

    # Create manager
    claude_manager = BedrockClaudeManager(region_name="us-east-1")

    try:
        # Run the complete workflow
        logger.info("Starting Claude model comparison...")

        results = claude_manager.run_inference_workflow(
            model_configs=model_configs,
            conversation_messages=conversation_messages,
            use_provisioned=False,  # Enable provisioned throughput usage
            streaming=False,
        )

        # Print final results summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)

        for model_id, result in results.items():
            print(f"\nModel: {model_id}")
            print(f"Status: {result['status']}")
            if result["status"] == "success":
                response_text = (
                    result["response"]["content"][0]["text"]
                    if "content" in result["response"]
                    else str(result["response"])
                )
                print(f"Response: {response_text[:200]}...")
                if "usage" in result:
                    print(f"Token Usage: {result['usage']}")
            else:
                print(f"Error: {result['error']}")
            print("-" * 40)

        return 0

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
