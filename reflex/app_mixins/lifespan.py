from __future__ import annotations
import asyncio
import contextlib
import functools
import inspect
from typing import Callable, Coroutine, Set, Union
from urllib.parse import urlparse

from fastapi import FastAPI
from starlette.types import ASGIApp

from reflex.config import get_config
from reflex.utils import console
from reflex.utils.exceptions import InvalidLifespanTaskType
from asgiproxy.config import BaseURLProxyConfigMixin, ProxyConfig
from asgiproxy.context import ProxyContext
from asgiproxy.simple_proxy import make_simple_proxy_app

from .mixin import AppMixin


class LifespanMixin(AppMixin):
    """A mixin that allows tasks to run during the app's lifespan, including proxy support."""

    # Lifespan tasks that are planned to run.
    lifespan_tasks: Set[Union[asyncio.Task, Callable]] = set()

    @contextlib.asynccontextmanager
    async def _run_lifespan_tasks(self, app: FastAPI):
        running_tasks = []
        try:
            async with contextlib.AsyncExitStack() as stack:
                # Enter the proxy middleware context
                await stack.enter_async_context(self.proxy_middleware(app))

                for task in self.lifespan_tasks:
                    run_msg = f"Started lifespan task: {task.__name__} as {{type}}"  # type: ignore
                    if isinstance(task, asyncio.Task):
                        running_tasks.append(task)
                    else:
                        signature = inspect.signature(task)
                        if "app" in signature.parameters:
                            task = functools.partial(task, app=app)
                        _t = task()
                        if isinstance(_t, contextlib.AbstractAsyncContextManager):
                            await stack.enter_async_context(_t)
                            console.debug(run_msg.format(type="asynccontextmanager"))
                        elif asyncio.iscoroutine(_t):
                            task_ = asyncio.create_task(_t)
                            task_.add_done_callback(lambda t: t.result())
                            running_tasks.append(task_)
                            console.debug(run_msg.format(type="coroutine"))
                        else:
                            console.debug(run_msg.format(type="function"))
                yield
        finally:
            for task in running_tasks:
                console.debug(f"Canceling lifespan task: {task}")
                task.cancel()

    def register_lifespan_task(self, task: Callable | asyncio.Task, **task_kwargs):
        """Register a task to run during the lifespan of the app.

        Args:
            task: The task to register.
            task_kwargs: The kwargs of the task.

        Raises:
            InvalidLifespanTaskType: If the task is a generator function.
        """
        if inspect.isgeneratorfunction(task) or inspect.isasyncgenfunction(task):
            raise InvalidLifespanTaskType(
                f"Task {task.__name__} of type generator must be decorated with contextlib.asynccontextmanager."
            )

        if task_kwargs:
            original_task = task
            task = functools.partial(task, **task_kwargs)  # type: ignore
            functools.update_wrapper(task, original_task)  # type: ignore
        self.lifespan_tasks.add(task)  # type: ignore
        console.debug(f"Registered lifespan task: {task.__name__}")  # type: ignore

    @contextlib.asynccontextmanager
    async def proxy_middleware(self, app: FastAPI):
        """Middleware to proxy requests to the separate frontend server.

        Args:
            app: The FastAPI instance.

        Yields:
            None
        """
        # Retrieve backend and frontend ports from the configuration
        config = get_config()
        backend_port = config.backend_port
        frontend_port = config.frontend_port
        frontend_host = f"http://localhost:{frontend_port}"

        class LocalProxyConfig(BaseURLProxyConfigMixin, ProxyConfig):
            upstream_base_url = frontend_host
            rewrite_host_header = urlparse(upstream_base_url).netloc

        proxy_context = ProxyContext(LocalProxyConfig())
        proxy_app = make_simple_proxy_app(proxy_context)

        # Mount the proxy app at the root path
        app.mount("/", proxy_app)

        console.debug(
            f"Proxying '/' requests on port {backend_port} to {frontend_host}"
        )
        async with proxy_context:
            yield
