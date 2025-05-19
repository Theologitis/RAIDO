from flwr.client.message_handler.message_handler import (
    handle_legacy_message_from_msgtype,
)
from flwr.client.mod.utils import make_ffn
from flwr.common.logger import warn_deprecated_feature
from flwr.client.typing import ClientFnExt, Mod
import inspect
from flwr.client.client import Client 
from flwr.common import Context
from omegaconf import DictConfig
from flwr.client.mod import fixedclipping_mod, secaggplus_mod ,adaptiveclipping_mod
from flowerapp.utils import drop_empty_keys
from flwr.common.config import unflatten_dict
from flwr.client import ClientApp

def _alert_erroneous_client_fn() -> None:
    raise ValueError(
        "A `ClientApp` cannot make use of a `client_fn` that does "
        "not have a signature in the form: `def client_fn(context: "
        "Context)`. You can import the `Context` like this: "
        "`from flwr.common import Context`"
    )
def _inspect_maybe_adapt_client_fn_signature(client_fn: ClientFnExt) -> ClientFnExt:
    client_fn_args = inspect.signature(client_fn).parameters

    if len(client_fn_args) != 1:
        _alert_erroneous_client_fn()

    first_arg = list(client_fn_args.keys())[0]
    first_arg_type = client_fn_args[first_arg].annotation

    if first_arg_type is str or first_arg == "cid":
        # Warn previous signature for `client_fn` seems to be used
        warn_deprecated_feature(
            "`client_fn` now expects a signature `def client_fn(context: Context)`."
            "The provided `client_fn` has signature: "
            f"{dict(client_fn_args.items())}. You can import the `Context` like this:"
            " `from flwr.common import Context`"
        )

        # Wrap depcreated client_fn inside a function with the expected signature
        def adaptor_fn(
            context: Context,
        ) -> Client:  # pylint: disable=unused-argument
            # if patition-id is defined, pass it. Else pass node_id that should
            # always be defined during Context init.
            cid = context.node_config.get("partition-id", context.node_id)
            return client_fn(str(cid))  # type: ignore

        return adaptor_fn

    return client_fn

class MyClientApp(ClientApp):
    def __init__(self, client_fn=None, mods=None):

        super().__init__(client_fn=client_fn, mods=mods)
        
        # Ensure that _client_fn is set when using the base class constructor
        if client_fn is not None:
            self._client_fn = _inspect_maybe_adapt_client_fn_signature(client_fn)
        else:
            self._client_fn = None

    def _set_mods(self, context: Context):
        """ Set the mods dynamically based on context and configuration. """
        

        # Dynamically determine mods based on context and configuration
        cfg = DictConfig(drop_empty_keys(unflatten_dict(context.run_config)))  # Use context to get the run_config
        mods = []

        # Determine Differential Privacy mod (if any)
        dp_mod = None
        dp = cfg.dp
        if dp.flag.lower() == "true" and dp.side == "client":
            if dp.type == "fixed":
                dp_mod = fixedclipping_mod
            else:
                dp_mod = adaptiveclipping_mod

        # Determine Secure Aggregation mod (if any)
        secagg_mod = None
        if cfg.secagg.flag.lower() == "true":
            print("secagg added")
            secagg_mod = secaggplus_mod

        # Enforce correct order: SecAgg first, then DP otherwise it breaks
        if secagg_mod:
            mods.append(secagg_mod)
        if dp_mod:
            mods.append(dp_mod)


        # Update self._mods
        self._mods = mods

        # Re-wrap the client_fn if mods have changed
        if self._client_fn:
            self._call = make_ffn(
                lambda msg, ctx: handle_legacy_message_from_msgtype(
                    client_fn=self._client_fn, message=msg, context=ctx
                ),
                self._mods,
            )

    def __call__(self, message, context: Context):
        """ Override to inject custom logic and modify mods based on context. """
        # Custom logic for setting mods based on context
        self._set_mods(context)
        
        # Call the original __call__ method after updating mods
        return self._call(message, context)