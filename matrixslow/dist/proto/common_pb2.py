# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63ommon.proto\"\'\n\x04Node\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tnode_type\x18\x02 \x01(\t\"$\n\x06Matrix\x12\r\n\x05value\x18\x01 \x03(\x02\x12\x0b\n\x03\x64im\x18\x02 \x03(\x05\"Q\n\rNodeGradients\x12\x14\n\x05nodes\x18\x01 \x03(\x0b\x32\x05.Node\x12\x1a\n\tgradients\x18\x02 \x03(\x0b\x32\x07.Matrix\x12\x0e\n\x06\x61\x63\x63_no\x18\x03 \x01(\x05\"H\n\x16VariableWeightsReqResp\x12\x14\n\x05nodes\x18\x01 \x03(\x0b\x32\x05.Node\x12\x18\n\x07weights\x18\x02 \x03(\x0b\x32\x07.Matrixb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'common_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _NODE._serialized_start=16
  _NODE._serialized_end=55
  _MATRIX._serialized_start=57
  _MATRIX._serialized_end=93
  _NODEGRADIENTS._serialized_start=95
  _NODEGRADIENTS._serialized_end=176
  _VARIABLEWEIGHTSREQRESP._serialized_start=178
  _VARIABLEWEIGHTSREQRESP._serialized_end=250
# @@protoc_insertion_point(module_scope)
