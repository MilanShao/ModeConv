from ModeConvModel.models import LuxModeConvFastModelPL, LuxModeConvLaplaceModelPL
from ModeConvModel.models import LuxChebGNNModelPL
from ModeConvModel.models import LuxAGCRNModelPL
from ModeConvModel.models import LuxMtGNNModelPL
from ModeConvModel.models import SimulatedSmartBridgeModeConvFastModelPL
from ModeConvModel.models import SimulatedSmartBridgeAGCRNModelPL
from ModeConvModel.models import SimulatedSmartBridgeMtGNNModelPL
from ModeConvModel.models import SimulatedSmartBridgeChebGNNModelPL
from ModeConvModel.models import SimulatedSmartBridgeModeConvLaplaceModelPL


def select_model(args):
    if args.dataset == "luxemburg":
        if args.model == "ModeConvFast":
            return LuxModeConvFastModelPL(26, args)
        elif args.model == "ModeConvLaplace":
            return LuxModeConvLaplaceModelPL(26, args)
        elif args.model == "ChebConv":
            return LuxChebGNNModelPL(26, args)
        elif args.model == "AGCRN":
            return LuxAGCRNModelPL(26, args)
        elif args.model == "MtGNN":
            return LuxMtGNNModelPL(26, args)
        else:
            raise NotImplementedError(f"model {args.model} does not exist")
    elif args.dataset == "simulated_smart_bridge":
        if args.model == "ModeConvFast":
            return SimulatedSmartBridgeModeConvFastModelPL(12, args)
        elif args.model == "ModeConvLaplace":
            return SimulatedSmartBridgeModeConvLaplaceModelPL(12, args)
        elif args.model == "ChebConv":
            return SimulatedSmartBridgeChebGNNModelPL(12, args)
        elif args.model == "AGCRN":
            return SimulatedSmartBridgeAGCRNModelPL(12, args)
        elif args.model == "MtGNN":
            return SimulatedSmartBridgeMtGNNModelPL(12, args)
        else:
            raise NotImplementedError(f"model {args.model} does not exist")
    else:
        raise NotImplementedError(f"dataset {args.dataset} does not exist")