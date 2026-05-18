import asyncio


async def interruptible_async_generator(async_generator, interrupt_event):
    """
    Wraps an async generator to make it interruptible.
    """
    gen = async_generator.__aiter__()
    interrupt_task = asyncio.create_task(interrupt_event.wait())

    while True:
        next_task = asyncio.create_task(gen.__anext__())
        done, pending = await asyncio.wait(
            {next_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if interrupt_task in done:
            next_task.cancel()
            try:
                await next_task
            except asyncio.CancelledError:
                pass
            break

        if next_task in done:
            try:
                yield next_task.result()
            except StopAsyncIteration:
                break

    interrupt_task.cancel()
    try:
        await interrupt_task
    except asyncio.CancelledError:
        pass


def is_active(task):
    if not task or task.done() or task.cancelled():
        return False

    return True


async def interruptible(coroutine, interrupt_event):
    """
    Runs a coroutine and allows it to be interrupted by an asyncio.Event.

    Args:
        coroutine: The coroutine to run.
        interrupt_event: The asyncio.Event that signals an interruption.

    Returns:
        A tuple of (result, interrupted).
        - If not interrupted: (coroutine_result, False)
        - If interrupted: (None, True)
    """
    main_task = asyncio.create_task(coroutine)
    interrupt_task = asyncio.create_task(interrupt_event.wait())

    done, pending = await asyncio.wait(
        {main_task, interrupt_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    if interrupt_task in done:
        return None, True

    try:
        return main_task.result(), False
    except asyncio.CancelledError:
        return None, True
