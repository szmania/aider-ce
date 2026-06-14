from .utils.base_command import BaseCommand


class CompactCommand(BaseCommand):
    NORM_NAME = "compact"
    DESCRIPTION = "Force compaction of the chat history context"
    show_completion_notification = True

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        # Pass args as message parameter if it's not empty
        message = args.strip() if args else ""
        await coder.compact_context_if_needed(force=True, message=message)

    @classmethod
    def get_help(cls) -> str:
        return "Force compaction of the chat history context."
