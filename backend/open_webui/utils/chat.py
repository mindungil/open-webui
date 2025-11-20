import time
import logging
import sys

from aiocache import cached
from typing import Any, Optional
import random
import json
import inspect
import uuid
import asyncio
import aiohttp

from fastapi import Request, status
from starlette.responses import Response, StreamingResponse, JSONResponse


from open_webui.models.users import UserModel

from open_webui.socket.main import (
    sio,
    get_event_call,
    get_event_emitter,
)
from open_webui.functions import generate_function_chat_completion

from open_webui.routers.openai import (
    generate_chat_completion as generate_openai_chat_completion,
)

from open_webui.routers.ollama import (
    generate_chat_completion as generate_ollama_chat_completion,
)

from open_webui.routers.pipelines import (
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)

from open_webui.models.functions import Functions
from open_webui.models.models import Models


from open_webui.utils.plugin import (
    load_function_module_by_id,
    get_function_module_from_cache,
)
from open_webui.utils.models import get_all_models, check_model_access
from open_webui.utils.payload import convert_payload_openai_to_ollama
from open_webui.utils.response import (
    convert_response_ollama_to_openai,
    convert_streaming_response_ollama_to_openai,
)
from open_webui.utils.filter import (
    get_sorted_filter_ids,
    process_filter_functions,
)
from open_webui.utils.files import (
    get_document_url_from_base64,
)

from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL, BYPASS_MODEL_ACCESS_CONTROL


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def generate_direct_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    models: dict,
):
    log.info("generate_direct_chat_completion")

    metadata = form_data.pop("metadata", {})

    user_id = metadata.get("user_id")
    session_id = metadata.get("session_id")
    request_id = str(uuid.uuid4())  # Generate a unique request ID

    event_caller = get_event_call(metadata)

    channel = f"{user_id}:{session_id}:{request_id}"
    logging.info(f"WebSocket channel: {channel}")

    if form_data.get("stream"):
        q = asyncio.Queue()

        async def message_listener(sid, data):
            """
            Handle received socket messages and push them into the queue.
            """
            await q.put(data)

        # Register the listener
        sio.on(channel, message_listener)

        # Start processing chat completion in background
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )

        log.info(f"res: {res}")

        if res.get("status", False):
            # Define a generator to stream responses
            async def event_generator():
                nonlocal q
                try:
                    while True:
                        data = await q.get()  # Wait for new messages
                        if isinstance(data, dict):
                            if "done" in data and data["done"]:
                                break  # Stop streaming when 'done' is received

                            yield f"data: {json.dumps(data)}\n\n"
                        elif isinstance(data, str):
                            if "data:" in data:
                                yield f"{data}\n\n"
                            else:
                                yield f"data: {data}\n\n"
                except Exception as e:
                    log.debug(f"Error in event generator: {e}")
                    pass

            # Define a background task to run the event generator
            async def background():
                try:
                    del sio.handlers["/"][channel]
                except Exception as e:
                    pass

            # Return the streaming response
            return StreamingResponse(
                event_generator(), media_type="text/event-stream", background=background
            )
        else:
            raise Exception(str(res))
    else:
        res = await event_caller(
            {
                "type": "request:chat:completion",
                "data": {
                    "form_data": form_data,
                    "model": models[form_data["model"]],
                    "channel": channel,
                    "session_id": session_id,
                },
            }
        )

        if "error" in res and res["error"]:
            raise Exception(res["error"])

        return res


async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user: Any,
    bypass_filter: bool = False,
):
    log.debug(f"generate_chat_completion: {form_data}")
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    if hasattr(request.state, "metadata"):
        if "metadata" not in form_data:
            form_data["metadata"] = request.state.metadata
        else:
            form_data["metadata"] = {
                **form_data["metadata"],
                **request.state.metadata,
            }

    # GPT Template 외부 API 연동
    metadata = form_data.get("metadata", {})
    template_api_url = metadata.get("template_api_url")
    log.info(f"GPT Template API check - template_api_url: {template_api_url}")
    if template_api_url:
        # api_url에 전체 엔드포인트가 포함되어 있음 (예: http://example.com:8080/v1)
        # /chat/completions를 추가
        api_endpoint = f"{template_api_url.rstrip('/')}/chat/completions"

        # 외부 API로 전송할 페이로드 준비 (metadata 제외)
        external_payload = {
            "model": form_data.get("model"),
            "messages": form_data.get("messages", []),
            "stream": form_data.get("stream", False),
        }

        # 선택적 파라미터 추가
        if "temperature" in form_data:
            external_payload["temperature"] = form_data["temperature"]
        if "max_tokens" in form_data:
            external_payload["max_tokens"] = form_data["max_tokens"]
        if "top_p" in form_data:
            external_payload["top_p"] = form_data["top_p"]
        if "files" in form_data:
            external_payload["files"] = form_data["files"]
        if "tools" in form_data:
            external_payload["tools"] = form_data["tools"]

        # Helper function to process files in response
        def process_response_files(result_data, req, usr, chat_meta):
            """Process files in the response and upload them to storage"""
            processed_files = []

            # Check for files in the response
            # Expected format: {"files": [{"name": "file.pdf", "data": "base64...", "type": "application/pdf"}]}
            if "files" in result_data:
                for file_info in result_data.get("files", []):
                    try:
                        file_data = file_info.get("data", "")
                        file_name = file_info.get("name", "document")
                        file_type = file_info.get("type", "application/octet-stream")

                        # Create base64 string with proper format
                        if not file_data.startswith("data:"):
                            base64_string = f"data:{file_type};base64,{file_data}"
                        else:
                            base64_string = file_data

                        # Upload file and get URL
                        file_url, file_id = get_document_url_from_base64(
                            req,
                            base64_string,
                            chat_meta,
                            usr,
                            file_name
                        )

                        if file_url:
                            processed_files.append({
                                "type": "file",
                                "name": file_name,
                                "url": file_url,
                                "id": file_id,
                                "collection_name": ""
                            })
                            log.info(f"Processed file: {file_name} -> {file_url}")
                    except Exception as e:
                        log.error(f"Error processing file {file_info.get('name', 'unknown')}: {e}")

            return processed_files

        try:
            if form_data.get("stream", False):
                # 스트리밍 응답 처리 - 세션을 generator 내부에서 관리
                async def stream_external_response():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                api_endpoint,
                                json=external_payload,
                                headers={"Content-Type": "application/json"}
                            ) as response:
                                buffer = ""
                                async for line in response.content:
                                    if line:
                                        decoded_line = line.decode('utf-8', errors='replace')
                                        buffer += decoded_line

                                        # Check for complete SSE messages
                                        while "\n" in buffer:
                                            line_end = buffer.index("\n")
                                            complete_line = buffer[:line_end]
                                            buffer = buffer[line_end + 1:]

                                            if complete_line.startswith("data: "):
                                                data_content = complete_line[6:]
                                                if data_content == "[DONE]":
                                                    yield f"data: [DONE]\n\n".encode()
                                                    continue

                                                try:
                                                    json_data = json.loads(data_content)

                                                    # Check for files in streaming response
                                                    if "files" in json_data:
                                                        file_meta = {
                                                            "chat_id": metadata.get("chat_id", None),
                                                            "message_id": metadata.get("message_id", None),
                                                            "session_id": metadata.get("session_id", None),
                                                        }
                                                        processed_files = process_response_files(json_data, request, user, file_meta)
                                                        if processed_files:
                                                            # Emit files event
                                                            file_event = {
                                                                "choices": [{
                                                                    "delta": {
                                                                        "content": "",
                                                                        "files": processed_files
                                                                    }
                                                                }]
                                                            }
                                                            yield f"data: {json.dumps(file_event)}\n\n".encode()
                                                        # Remove files from original response
                                                        del json_data["files"]

                                                    yield f"data: {json.dumps(json_data)}\n\n".encode()
                                                except json.JSONDecodeError:
                                                    yield f"{complete_line}\n".encode()
                                            elif complete_line.strip():
                                                yield f"{complete_line}\n".encode()

                                # Yield remaining buffer
                                if buffer.strip():
                                    yield buffer.encode()
                    except Exception as e:
                        log.error(f"Error in streaming from {api_endpoint}: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

                return StreamingResponse(
                    stream_external_response(),
                    media_type="text/event-stream"
                )
            else:
                # 일반 응답 처리
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_endpoint,
                        json=external_payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        result = await response.json()

                        # Process files in non-streaming response
                        if "files" in result:
                            file_meta = {
                                "chat_id": metadata.get("chat_id", None),
                                "message_id": metadata.get("message_id", None),
                                "session_id": metadata.get("session_id", None),
                            }
                            processed_files = process_response_files(result, request, user, file_meta)
                            if processed_files:
                                # Add processed files to response
                                if "choices" in result and len(result["choices"]) > 0:
                                    if "message" not in result["choices"][0]:
                                        result["choices"][0]["message"] = {}
                                    result["choices"][0]["message"]["files"] = processed_files
                            # Remove raw file data from response
                            del result["files"]

                        return JSONResponse(content=result)
        except Exception as e:
            log.error(f"Error calling external API {api_endpoint}: {e}")
            raise Exception(f"External API call failed: {str(e)}")

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
        log.debug(f"direct connection to model: {models}")
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise Exception("Model not found")

    model = models[model_id]

    if getattr(request.state, "direct", False):
        return await generate_direct_chat_completion(
            request, form_data, user=user, models=models
        )
    else:
        # Check if user has access to the model
        if not bypass_filter and user.role == "user":
            try:
                check_model_access(user, model)
            except Exception as e:
                raise e

        if model.get("owned_by") == "arena":
            model_ids = model.get("info", {}).get("meta", {}).get("model_ids")
            filter_mode = model.get("info", {}).get("meta", {}).get("filter_mode")
            if model_ids and filter_mode == "exclude":
                model_ids = [
                    model["id"]
                    for model in list(request.app.state.MODELS.values())
                    if model.get("owned_by") != "arena" and model["id"] not in model_ids
                ]

            selected_model_id = None
            if isinstance(model_ids, list) and model_ids:
                selected_model_id = random.choice(model_ids)
            else:
                model_ids = [
                    model["id"]
                    for model in list(request.app.state.MODELS.values())
                    if model.get("owned_by") != "arena"
                ]
                selected_model_id = random.choice(model_ids)

            form_data["model"] = selected_model_id

            if form_data.get("stream") == True:

                async def stream_wrapper(stream):
                    yield f"data: {json.dumps({'selected_model_id': selected_model_id})}\n\n"
                    async for chunk in stream:
                        yield chunk

                response = await generate_chat_completion(
                    request, form_data, user, bypass_filter=True
                )
                return StreamingResponse(
                    stream_wrapper(response.body_iterator),
                    media_type="text/event-stream",
                    background=response.background,
                )
            else:
                return {
                    **(
                        await generate_chat_completion(
                            request, form_data, user, bypass_filter=True
                        )
                    ),
                    "selected_model_id": selected_model_id,
                }

        if model.get("pipe"):
            # Below does not require bypass_filter because this is the only route the uses this function and it is already bypassing the filter
            return await generate_function_chat_completion(
                request, form_data, user=user, models=models
            )
        if model.get("owned_by") == "ollama":
            # Using /ollama/api/chat endpoint
            form_data = convert_payload_openai_to_ollama(form_data)
            response = await generate_ollama_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
                bypass_filter=bypass_filter,
            )
            if form_data.get("stream"):
                response.headers["content-type"] = "text/event-stream"
                return StreamingResponse(
                    convert_streaming_response_ollama_to_openai(response),
                    headers=dict(response.headers),
                    background=response.background,
                )
            else:
                return convert_response_ollama_to_openai(response)
        else:
            return await generate_openai_chat_completion(
                request=request,
                form_data=form_data,
                user=user,
                bypass_filter=bypass_filter,
            )


chat_completion = generate_chat_completion


async def chat_completed(request: Request, form_data: dict, user: Any):
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    data = form_data
    model_id = data["model"]
    if model_id not in models:
        raise Exception("Model not found")

    model = models[model_id]

    try:
        data = await process_pipeline_outlet_filter(request, data, user, models)
    except Exception as e:
        return Exception(f"Error: {e}")

    metadata = {
        "chat_id": data["chat_id"],
        "message_id": data["id"],
        "filter_ids": data.get("filter_ids", []),
        "session_id": data["session_id"],
        "user_id": user.id,
    }

    extra_params = {
        "__event_emitter__": get_event_emitter(metadata),
        "__event_call__": get_event_call(metadata),
        "__user__": user.model_dump() if isinstance(user, UserModel) else {},
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }

    try:
        filter_functions = [
            Functions.get_function_by_id(filter_id)
            for filter_id in get_sorted_filter_ids(
                request, model, metadata.get("filter_ids", [])
            )
        ]

        result, _ = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="outlet",
            form_data=data,
            extra_params=extra_params,
        )
        return result
    except Exception as e:
        return Exception(f"Error: {e}")


async def chat_action(request: Request, action_id: str, form_data: dict, user: Any):
    if "." in action_id:
        action_id, sub_action_id = action_id.split(".")
    else:
        sub_action_id = None

    action = Functions.get_function_by_id(action_id)
    if not action:
        raise Exception(f"Action not found: {action_id}")

    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    data = form_data
    model_id = data["model"]

    if model_id not in models:
        raise Exception("Model not found")
    model = models[model_id]

    __event_emitter__ = get_event_emitter(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )
    __event_call__ = get_event_call(
        {
            "chat_id": data["chat_id"],
            "message_id": data["id"],
            "session_id": data["session_id"],
            "user_id": user.id,
        }
    )

    function_module, _, _ = get_function_module_from_cache(request, action_id)

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        valves = Functions.get_function_valves_by_id(action_id)
        function_module.valves = function_module.Valves(**(valves if valves else {}))

    if hasattr(function_module, "action"):
        try:
            action = function_module.action

            # Get the signature of the function
            sig = inspect.signature(action)
            params = {"body": data}

            # Extra parameters to be passed to the function
            extra_params = {
                "__model__": model,
                "__id__": sub_action_id if sub_action_id is not None else action_id,
                "__event_emitter__": __event_emitter__,
                "__event_call__": __event_call__,
                "__request__": request,
            }

            # Add extra params in contained in function signature
            for key, value in extra_params.items():
                if key in sig.parameters:
                    params[key] = value

            if "__user__" in sig.parameters:
                __user__ = user.model_dump() if isinstance(user, UserModel) else {}

                try:
                    if hasattr(function_module, "UserValves"):
                        __user__["valves"] = function_module.UserValves(
                            **Functions.get_user_valves_by_id_and_user_id(
                                action_id, user.id
                            )
                        )
                except Exception as e:
                    log.exception(f"Failed to get user values: {e}")

                params = {**params, "__user__": __user__}

            if inspect.iscoroutinefunction(action):
                data = await action(**params)
            else:
                data = action(**params)

        except Exception as e:
            return Exception(f"Error: {e}")

    return data
