# Copyright 2025 github.com/cerebraljam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TFRecord Inspector for Facade
Utility to examine the structure of binary TFRecord files used in Facade.

Based on Google's Facade project architecture.
"""

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.example import example_pb2
import argparse


def inspect_tfrecord_file(filename: str, max_examples: int = 5) -> None:
    """
    Inspect a TFRecord file and print its contents in human-readable format.
    
    Args:
        filename: Path to the TFRecord file
        max_examples: Maximum number of examples to display
    """
    print(f"Inspecting TFRecord file: {filename}")
    print("=" * 60)
    
    dataset = tf.data.TFRecordDataset([filename])
    
    for i, serialized_example in enumerate(dataset.take(max_examples)):
        print(f"\n--- Example {i+1} ---")
        
        try:
            # Try parsing as SequenceExample (most common in Facade)
            example = example_pb2.SequenceExample()
            example.ParseFromString(serialized_example.numpy())
            
            print("Type: SequenceExample")
            print("Context features:")
            for key, feature in example.context.feature.items():
                print(f"  {key}: {_format_feature(feature)}")
            
            print("Sequence features:")
            for key, feature_list in example.feature_lists.feature_list.items():
                print(f"  {key}: {len(feature_list.feature)} timesteps")
                for j, feature in enumerate(feature_list.feature):
                    print(f"    [{j}]: {_format_feature(feature)}")
                    
        except Exception as e1:
            try:
                # Try parsing as regular Example
                example = example_pb2.Example()
                example.ParseFromString(serialized_example.numpy())
                
                print("Type: Example")
                for key, feature in example.features.feature.items():
                    print(f"  {key}: {_format_feature(feature)}")
                    
            except Exception as e2:
                print(f"Failed to parse as SequenceExample: {e1}")
                print(f"Failed to parse as Example: {e2}")
                print("Raw bytes length:", len(serialized_example.numpy()))


def _format_feature(feature) -> str:
    """Format a TensorFlow Feature for display."""
    if feature.HasField('bytes_list'):
        values = [v.decode('utf-8', errors='replace') for v in feature.bytes_list.value]
        return f"bytes: {values}"
    elif feature.HasField('float_list'):
        return f"floats: {list(feature.float_list.value)}"
    elif feature.HasField('int64_list'):
        return f"ints: {list(feature.int64_list.value)}"
    else:
        return "empty"


def convert_textproto_to_tfrecord(textproto_file: str, output_file: str) -> None:
    """
    Convert a textproto file of SequenceExamples to binary TFRecord format.
    
    Args:
        textproto_file: Path to textproto file with SequenceExample messages
        output_file: Path for output TFRecord file
    """
    print(f"Converting {textproto_file} to {output_file}")
    
    with tf.io.TFRecordWriter(output_file) as writer:
        with open(textproto_file, 'r') as reader:
            txt = ''
            for line in reader:
                if line.strip() == '':
                    if txt.strip():
                        try:
                            example = text_format.Parse(txt, example_pb2.SequenceExample())
                            writer.write(example.SerializeToString())
                            print("Converted one SequenceExample")
                        except Exception as e:
                            print(f"Failed to parse example: {e}")
                            print(f"Text was: {txt[:200]}...")
                    txt = ''
                else:
                    txt += line
            
            # Handle last example if file doesn't end with blank line
            if txt.strip():
                try:
                    example = text_format.Parse(txt, example_pb2.SequenceExample())
                    writer.write(example.SerializeToString())
                    print("Converted final SequenceExample")
                except Exception as e:
                    print(f"Failed to parse final example: {e}")


def main():
    parser = argparse.ArgumentParser(description='Inspect or convert Facade TFRecord files')
    parser.add_argument('command', choices=['inspect', 'convert'], 
                       help='Command to execute')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('--output', help='Output file path (for convert command)')
    parser.add_argument('--max_examples', type=int, default=5, 
                       help='Maximum examples to display (for inspect command)')
    
    args = parser.parse_args()
    
    if args.command == 'inspect':
        inspect_tfrecord_file(args.input_file, args.max_examples)
    elif args.command == 'convert':
        if not args.output:
            print("Error: --output required for convert command")
            return
        convert_textproto_to_tfrecord(args.input_file, args.output)


if __name__ == '__main__':
    main()