import re
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple


class ExtractWorld:
    """Class for extracting and transforming objects from Webots world files"""

    def __init__(self, path):
        """Initialize with a path to a world file"""
        self.world_file_path = path
        self.control_points = []

    def extract_gate_data(self, content: str) -> List[Dict]:
        """
        Extract complete gate data from VRML content, including beam subfields
        Returns list of dictionaries containing gate properties
        """
        gate_pattern = re.compile(
            r'DEF (\w+)\s+RacingGate\s*{(.*?)}',
            re.DOTALL
        )

        beam_fields = [
            'topBeamTranslation', 'topBeamScale',
            'bottomBeamTranslation', 'bottomBeamScale',
            'leftBeamTranslation', 'leftBeamScale',
            'rightBeamTranslation', 'rightBeamScale',
            'leftLegTranslation', 'rightLegTranslation'
        ]

        gates = []

        for match in gate_pattern.finditer(content):
            gate_name = match.group(1)
            block = match.group(2)

            gate = {
                'type': 'gate',
                'name': gate_name,
                'raw_data': match.group(0)
            }

            # Extract basic fields
            translation_match = re.search(r'translation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block)
            rotation_match = re.search(r'rotation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)',
                                       block)
            goal_size_match = re.search(r'goalSize\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block)

            if translation_match:
                gate['translation'] = list(map(float, translation_match.groups()))
            if rotation_match:
                gate['rotation'] = list(map(float, rotation_match.groups()))
            if goal_size_match:
                gate['scale'] = list(map(float, goal_size_match.groups()))

            # Extract all beam fields
            for key in beam_fields:
                match = re.search(fr'{key}\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block)
                if match:
                    gate[key] = list(map(float, match.groups()))

            gates.append(gate)

        return gates

    def extract_takeoff_pad(self, content: str) -> List[Dict]:
        """Extract raw takeoff pad data from VRML content"""
        takeoff_pad_pattern = re.compile(
            r'DEF\s+TAKE_OFF_PAD\s+Solid\s*{[^}]*?'
            r'translation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s*'
            r'rotation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+).*?'
            r'geometry\s+Box\s*{\s*size\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)',
            re.DOTALL
        )

        return [
            {
                'type': 'takeoff_pad',
                'name': 'TAKE_OFF_PAD',
                'translation': list(map(float, match.groups()[0:3])),
                'rotation': list(map(float, match.groups()[3:7])),
                'scale': list(map(float, match.groups()[7:10])),
                'raw_data': match.group(0)
            }
            for match in takeoff_pad_pattern.finditer(content)
        ]

    def extract_beams(self, content: str) -> List[Dict]:
        """Extract beam data from RacingGate blocks"""
        gate_pattern = re.compile(
            r'DEF (\w+)\s+RacingGate\s*{.*?}',
            re.DOTALL
        )

        beam_names = [
            ('topBeam', True),
            ('bottomBeam', True),
            ('leftBeam', True),
            ('rightBeam', True),
        ]

        beams = []

        for match in gate_pattern.finditer(content):
            block = match.group(0)

            # Get gate-level translation and rotation
            gate_translation = list(
                map(float, re.search(r'translation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block).groups()))
            gate_rotation = list(map(float, re.search(
                r'rotation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block).groups()))
            gate_rot = Rotation.from_rotvec(np.array(gate_rotation[:3]) * gate_rotation[3])

            for name, has_scale in beam_names:
                trans_match = re.search(fr'{name}Translation\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block)
                scale_match = re.search(fr'{name}Scale\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', block)

                if trans_match and scale_match:
                    local_translation = np.array(list(map(float, trans_match.groups())))
                    scale = list(map(float, scale_match.groups()))

                    # Transform beam local position to world position
                    world_translation = gate_rot.apply(local_translation) + gate_translation

                    beams.append({
                        'type': 'beam',
                        'name': f'{name}_{match.group(1)}',
                        'translation': world_translation.tolist(),
                        'rotation': gate_rotation,
                        'scale': scale,
                        'raw_data': block
                    })

        return beams

    def extract_obstacles(self, world_text):
        obstacle_pattern = re.compile(
            r'DEF\s+(OBSTACLE_\d+)\s+Solid\s*{(.*?)}',
            re.DOTALL
        )
        translation_pattern = re.compile(
            r'translation\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)'
        )
        rotation_pattern = re.compile(
            r'rotation\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)'
        )
        # This pattern will match: # size x y z (with any whitespace before/after)
        size_comment_pattern = re.compile(
            r'#\s*size\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)'
        )

        obstacles = []
        for match in obstacle_pattern.finditer(world_text):
            name = match.group(1)
            body = match.group(2)

            # print(f"\n--- Processing Obstacle: {name} ---")
            # print("[DEBUG] Obstacle body:\n", body[:200])  # preview body for debugging

            translation_match = translation_pattern.search(body)
            rotation_match = rotation_pattern.search(body)
            size_comment_match = size_comment_pattern.search(body)

            # print("[DEBUG] Translation match:", translation_match.group(0) if translation_match else "None")
            # print("[DEBUG] Rotation match:", rotation_match.group(0) if rotation_match else "None")
            # print("[DEBUG] Size comment match:", size_comment_match.group(0) if size_comment_match else "None")

            translation = tuple(map(float, translation_match.groups())) if translation_match else None
            rotation = tuple(map(float, rotation_match.groups())) if rotation_match else None
            scale = tuple(map(float, size_comment_match.groups())) if size_comment_match else None

            obstacles.append({
                'type': 'obstacle',
                'name': name,
                'translation': translation,
                'rotation': rotation,
                'scale': scale
            })

        # print("\n[RESULT] Total Obstacles Extracted:", len(obstacles))
        # for obstacle in obstacles:
        #     print(f"Obstacle Name: {obstacle['name']}")
        #     print(f"Translation: {obstacle['translation']}")
        #     print(f"Rotation: {obstacle['rotation']}")
        #     print(f"Scale: {obstacle['scale']}")
        #     print("----------------------------------------")

        return obstacles

    def extract_object_data(self) -> List[Dict]:
        """
        Main extraction function for gates and takeoff pads
        Returns combined list of all extracted objects
        """
        with open(self.world_file_path, 'r') as f:
            content = f.read()

        gate_data = self.extract_gate_data(content)
        takeoff_pad_data = self.extract_takeoff_pad(content)
        # beams_data = self.extract_beams(content)  # Uncomment when needed
        obstacles_data = self.extract_obstacles(content)
        # print(f"Total Obstacles Extracted: {len(obstacles_data)}")

        return gate_data + takeoff_pad_data + obstacles_data

    def transform_coordinates(self, objects: List[Dict]) -> List[Dict]:
        """
        Apply coordinate system transformations to all objects
        Handles gate+beam expansion automatically
        """
        transformed = []

        for obj in objects:
            if obj['type'] == 'gate':
                transformed.extend(self.transform_gate_and_beams(obj))
            else:
                transformed.append(self.transform_generic_object(obj))

        return transformed

    def transform_gate_and_beams(self, gate_obj: Dict) -> List[Dict]:
        """
        Transform gate and its beams to the world frame and coordinate system
        Returns list of transformed objects (gate + beams)
        """
        transformed = []

        # Transform the gate itself
        gate_transformed = self.transform_generic_object(gate_obj)
        transformed.append(gate_transformed)

        # Extract and transform beams
        beams = self.convert_gates_to_beam_objects([gate_obj])
        for beam in beams:
            transformed_beam = self.transform_generic_object(beam)
            transformed.append(transformed_beam)

        return transformed

    def transform_generic_object(self, obj: Dict) -> Dict:
        """
        Coordinate system transformation for a single object
        Applies offset and rotation adjustments
        """
        x, y, z = obj['translation']
        rx, ry, rz, angle = obj['rotation']

        # Apply offset (1m in x and y)
        new_x = x + 1
        new_y = y + 1
        new_z = z

        # Special handling for gates
        if obj['type'] == 'gate' or obj['type'] == 'obstacle':
            new_rotation = [0, 0, 1, angle]  # Simplified rotation
        else:
            # Leave the rotation unchanged for beams and other objects
            new_rotation = [rx, ry, rz, angle]

        return {
            **obj,
            'translation': [new_x, new_y, new_z],
            'rotation': new_rotation,
            'original_translation': obj['translation']  # Preserve original
        }

    def convert_gates_to_beam_objects(self, gates: List[Dict]) -> List[Dict]:
        """
        Extract and transform beam components from gate objects into world-space obstacles
        Returns list of beam objects with proper world coordinates
        """
        beam_objects = []

        for gate in gates:
            gate_pos = np.array(gate['translation'])
            gate_rotvec = np.array(gate['rotation'][:3]) * gate['rotation'][3]
            gate_rot = Rotation.from_rotvec(gate_rotvec)

            # print(f"\n=== Processing {gate['name']} ===")
            # print(f"  Gate Position: {gate_pos}")
            # print(f"  Gate Rotation vec: {gate_rotvec} (angle={np.linalg.norm(gate_rotvec)})")

            beam_defs = [
                ('top_beam', 'topBeamTranslation', 'topBeamScale'),
                ('bottom_beam', 'bottomBeamTranslation', 'bottomBeamScale'),
                ('left_beam', 'leftBeamTranslation', 'leftBeamScale'),
                ('right_beam', 'rightBeamTranslation', 'rightBeamScale'),
            ]

            for beam_type, trans_key, scale_key in beam_defs:
                if trans_key in gate and scale_key in gate:
                    local_pos = np.array(gate[trans_key])
                    scale = np.array(gate[scale_key])

                    world_pos, full_rotation = self.transform_relative_to_gate(
                        gate_pos, gate_rot, beam_type, local_pos, scale
                    )

                    beam_objects.append({
                        'type': 'beam',
                        'name': f"{gate['name']}_{beam_type}",
                        'translation': world_pos.tolist(),
                        'rotation': full_rotation,
                        'scale': scale.tolist()
                    })
                else:
                    print(f"  [SKIP] {beam_type}: Missing keys")

        return beam_objects

    def transform_relative_to_gate(self, gate_pos: np.ndarray, gate_rot: Rotation,
                                   beam_name: str, local_pos: np.ndarray, scale: np.ndarray) -> tuple:
        """
        Transform a beam's local position to world space using the gate's position and rotation
        Returns tuple of (world_position, full_rotation)
        """
        # Define beam-specific local rotations
        if 'top_beam' in beam_name or 'bottom_beam' in beam_name:
            local_rot = Rotation.from_euler('z', np.pi / 2)  # rotate to be parallel with gate
        elif 'left_beam' in beam_name or 'right_beam' in beam_name:
            local_rot = Rotation.from_euler('y', np.pi / 2)  # rotate to be coplanar with gate
        else:
            local_rot = Rotation.identity()

        # Apply local beam rotation first, then gate's rotation
        world_rot = gate_rot * local_rot

        # Apply gate rotation to translate position (only gate_rot, not local_rot!)
        world_pos = gate_rot.apply(local_pos) + gate_pos

        # Compose final rotation vector
        rotvec = world_rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = (rotvec / angle).tolist() if angle > 1e-6 else [0, 0, 1]
        full_rotation = axis + [angle]

        # print(f"\n    [{beam_name}]")
        # print(f"      Local Translation: {local_pos}")
        # print(f"      Local Scale: {scale}")
        # print(f"      → World Position: {world_pos}")
        # print(f"      → Final Rotation (axis + angle): {full_rotation}")

        return world_pos, full_rotation

    def get_control_points(self, objects: List[Dict]) -> Tuple[List[tuple], List[float]]:
        """
        Creates a list of control points from gate positions and landing pad.

        Args:
            objects: List of dictionaries containing object data with translations and types

        Returns:
        tuple:
            - List[tuple]: List of (x, y, z) tuples representing control points for the path
            - List[float]: List of rotation angles (theta) for each control point
        """
        control_points = []
        thetas = []
        landing_pad_pos = None

        # Sort gates by name to ensure consistent ordering
        gates = [obj for obj in objects if obj['type'] == 'gate']
        gates.sort(key=lambda x: x['name'])

        # Add gate positions as control points
        for gate in gates:
            pos = gate['translation']
            theta = gate['rotation'][3]  # Extract rotation angle
            control_points.append((pos[0], pos[1], pos[2]))
            thetas.append(theta)

        # Find landing pad position
        for obj in objects:
            if obj['type'] == 'takeoff_pad':
                landing_pad_pos = obj['translation']
                break

        if landing_pad_pos:
            # Add final point 1 meter above landing pad (theta=0 for landing pad)
            final_point = (landing_pad_pos[0], landing_pad_pos[1], landing_pad_pos[2] + 1.0)
            control_points.append(final_point)
            thetas.append(0)

        self.control_points = control_points

        return control_points, thetas

