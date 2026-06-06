import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box, Cylinder } from '@react-three/drei';
import * as THREE from 'three';
import { useNexusStore, DroneState, TeamName } from '../../store/nexus';

const TEAM_HEX: Record<TeamName, number> = {
  RED:   0xff4060,
  BLUE:  0x4080ff,
  GREEN: 0x00ff9d,
  GOLD:  0xffb800,
};

const SCALE = 0.01;  // world units → Three.js units

function toThree(x: number, y: number, z = 0): [number, number, number] {
  return [(x - 600) * SCALE, z, -(y - 450) * SCALE];  // center on origin
}

// ─── Drone mesh ──────────────────────────────────────────────────────────────

function DroneModel({ drone }: { drone: DroneState }) {
  const meshRef = useRef<THREE.Group>(null);
  const color = TEAM_HEX[drone.team] || 0xffffff;
  const isDead = drone.state === 'dead';
  const { selectDrone, selectedDroneId } = useNexusStore();
  const isSelected = drone.id === selectedDroneId;

  useFrame(() => {
    if (!meshRef.current) return;
    const [tx, , tz] = toThree(drone.position.x, drone.position.y);
    meshRef.current.position.x = tx;
    meshRef.current.position.z = tz;
    meshRef.current.position.y = isDead ? -0.05 : 0 + Math.sin(Date.now() * 0.002 + drone.id.charCodeAt(0)) * 0.02;
    meshRef.current.rotation.y = -drone.heading;
  });

  const healthRatio = drone.health / 100;

  return (
    <group ref={meshRef} onClick={() => selectDrone(isSelected ? null : drone.id)}>
      {/* Body */}
      <mesh castShadow>
        <octahedronGeometry args={[0.06, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isDead ? 0.02 : isSelected ? 0.8 : 0.3}
          transparent
          opacity={isDead ? 0.2 : 1.0}
          wireframe={isSelected}
        />
      </mesh>

      {/* Direction indicator */}
      {!isDead && (
        <mesh position={[0, 0, -0.09]}>
          <coneGeometry args={[0.015, 0.05, 4]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
        </mesh>
      )}

      {/* Health bar (billboard) */}
      {!isDead && (
        <group position={[0, 0.12, 0]}>
          <mesh position={[0, 0, 0]}>
            <planeGeometry args={[0.12, 0.012]} />
            <meshBasicMaterial color={0x111111} />
          </mesh>
          <mesh position={[-(0.06 - 0.06 * healthRatio), 0, 0.001]}>
            <planeGeometry args={[0.12 * healthRatio, 0.010]} />
            <meshBasicMaterial color={healthRatio > 0.5 ? 0x00ff9d : healthRatio > 0.25 ? 0xffb800 : 0xff4060} />
          </mesh>
        </group>
      )}

      {/* Glow point light for selected */}
      {isSelected && (
        <pointLight color={color} intensity={0.5} distance={0.5} />
      )}
    </group>
  );
}

// ─── Projectile ──────────────────────────────────────────────────────────────

function ProjectileMesh({ proj }: { proj: any }) {
  const color = TEAM_HEX[proj.team as TeamName] || 0xffffff;
  return (
    <mesh position={toThree(proj.x, proj.y, 0.02)}>
      <sphereGeometry args={[0.015, 4, 4]} />
      <meshBasicMaterial color={color} />
      <pointLight color={color} intensity={0.3} distance={0.2} />
    </mesh>
  );
}

// ─── Arena floor ─────────────────────────────────────────────────────────────

function ArenaFloor({ arena }: { arena: any }) {
  const w = arena.width * SCALE;
  const h = arena.height * SCALE;

  return (
    <group>
      {/* Base floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[w, h]} />
        <meshStandardMaterial color={0x060810} metalness={0.3} roughness={0.8} />
      </mesh>

      {/* Grid lines */}
      {Array.from({ length: 21 }, (_, i) => i * w / 20 - w / 2).map((x, i) => (
        <Line key={`gx${i}`}
          points={[[x, 0, -h / 2], [x, 0, h / 2]]}
          color={0x00d2ff} lineWidth={0.5} transparent opacity={0.06} />
      ))}
      {Array.from({ length: 16 }, (_, i) => i * h / 15 - h / 2).map((z, i) => (
        <Line key={`gz${i}`}
          points={[[-w / 2, 0, z], [w / 2, 0, z]]}
          color={0x00d2ff} lineWidth={0.5} transparent opacity={0.06} />
      ))}

      {/* Arena border */}
      <Line
        points={[[-w/2,0,-h/2],[ w/2,0,-h/2],[ w/2,0, h/2],[-w/2,0, h/2],[-w/2,0,-h/2]]}
        color={0x00d2ff} lineWidth={2} transparent opacity={0.6} />

      {/* Obstacles */}
      {arena.obstacles?.map((obs: any, i: number) => {
        const cx = (obs.x + obs.w / 2 - 600) * SCALE;
        const cz = -(obs.y + obs.h / 2 - 450) * SCALE;
        return (
          <mesh key={i} position={[cx, 0.04, cz]} castShadow>
            <boxGeometry args={[obs.w * SCALE, 0.08, obs.h * SCALE]} />
            <meshStandardMaterial color={0x0d1f2e} emissive={0x002030} emissiveIntensity={0.3} metalness={0.5} roughness={0.5} />
          </mesh>
        );
      })}

      {/* Control points */}
      {arena.control_points?.map((cp: any) => {
        const [cx, , cz] = toThree(cp.x, cp.y);
        const ownerColor = cp.owner ? TEAM_HEX[cp.owner as TeamName] : 0x00d2ff;
        return (
          <group key={cp.id} position={[cx, 0, cz]}>
            {/* Ring */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[cp.r * SCALE - 0.02, cp.r * SCALE, 32]} />
              <meshBasicMaterial color={ownerColor} transparent opacity={0.4} side={THREE.DoubleSide} />
            </mesh>
            {/* Capture fill */}
            {(cp.capture ?? 0) > 0 && (
              <pointLight color={ownerColor} intensity={cp.capture * 0.8} distance={cp.r * SCALE * 3} />
            )}
            <Text position={[0, 0.1, 0]} fontSize={0.04} color={0x00d2ff} anchorX="center">
              {cp.id.toUpperCase()}
            </Text>
          </group>
        );
      })}

      {/* Hazards */}
      {arena.hazards?.map((h: any) => {
        const [hx, , hz] = toThree(h.x, h.y);
        const hazardColors: Record<string, number> = {
          plasma_storm: 0xff5028, gravity_well: 0x8c28ff,
          emp_pulse: 0x00ffdc, shield_disrupt: 0xffc800,
        };
        const hColor = hazardColors[h.type] || 0xff0000;
        return (
          <group key={h.id} position={[hx, 0, hz]}>
            <pointLight color={hColor} intensity={h.intensity * 0.6} distance={h.radius * SCALE * 2} />
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <circleGeometry args={[h.radius * SCALE, 32]} />
              <meshBasicMaterial color={hColor} transparent opacity={0.08} side={THREE.DoubleSide} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// ─── Camera controls ──────────────────────────────────────────────────────────

function AutoCamera({ selectedDroneId }: { selectedDroneId: string | null }) {
  const { camera } = useThree();
  const drones = useNexusStore(s => s.drones);

  useFrame(() => {
    if (!selectedDroneId) return;
    const drone = drones.find(d => d.id === selectedDroneId);
    if (!drone) return;
    const [tx, , tz] = toThree(drone.position.x, drone.position.y);
    // Lerp camera target toward selected drone
    camera.position.lerp(new THREE.Vector3(tx + 0.3, 0.4, tz + 0.3), 0.02);
  });

  return null;
}

// ─── Main 3D view ─────────────────────────────────────────────────────────────

export function Arena3DView() {
  const { drones, projectiles, arena, selectedDroneId } = useNexusStore();

  return (
    <Canvas
      camera={{ position: [0.8, 0.9, 0.8], fov: 55, near: 0.01, far: 50 }}
      shadows
      gl={{ antialias: true }}
      style={{ background: '#060810' }}
    >
      <ambientLight intensity={0.15} />
      <directionalLight position={[2, 3, 2]} intensity={0.4} castShadow />
      <pointLight position={[0, 1, 0]} color={0x00d2ff} intensity={0.1} />

      <OrbitControls
        enableDamping dampingFactor={0.05}
        minDistance={0.3} maxDistance={3}
        maxPolarAngle={Math.PI / 2.1}
      />

      <AutoCamera selectedDroneId={selectedDroneId} />

      {arena && <ArenaFloor arena={arena} />}

      {drones.map(drone => (
        <DroneModel key={drone.id} drone={drone} />
      ))}

      {projectiles.map(proj => (
        <ProjectileMesh key={proj.id} proj={proj} />
      ))}

      {/* Fog for atmosphere */}
      <fog attach="fog" args={[0x060810, 3, 8]} />
    </Canvas>
  );
}
