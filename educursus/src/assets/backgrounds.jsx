import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

export const AuroraBackground = () => <div id="aurora-background"></div>;

const ThreeJSLandingBackgroundComponent = () => {
    const mountRef = useRef(null);

    useEffect(() => {
        if (!mountRef.current) return;
        let currentRef = mountRef.current;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(
            75,
            currentRef.clientWidth / currentRef.clientHeight,
            0.1,
            1000
        );
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(currentRef.clientWidth, currentRef.clientHeight);
        currentRef.appendChild(renderer.domElement);

        // Wireframe globe
        const geometry = new THREE.IcosahedronGeometry(3, 1);
        const wireframe = new THREE.WireframeGeometry(geometry);
        const line = new THREE.LineSegments(wireframe);
        line.material.color = new THREE.Color(0x00bfff);
        line.material.depthTest = false;
        line.material.opacity = 0.5;
        line.material.transparent = true;
        scene.add(line);

        // Particle background
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCnt = 5000;
        const posArray = new Float32Array(particlesCnt * 3);
        for (let i = 0; i < particlesCnt * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 25;
        }
        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        const particlesMaterial = new THREE.PointsMaterial({ size: 0.005, color: 0x00bfff });
        const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        scene.add(particlesMesh);

        camera.position.z = 10;

        // Mouse controls
        const mouse = new THREE.Vector2();
        const handleMouseMove = (event) => {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        };
        window.addEventListener('mousemove', handleMouseMove);

        const handleResize = () => {
            if (currentRef) {
                camera.aspect = currentRef.clientWidth / currentRef.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(currentRef.clientWidth, currentRef.clientHeight);
            }
        };
        window.addEventListener('resize', handleResize);

        let frameId;
        const clock = new THREE.Clock();
        const animate = () => {
            const elapsedTime = clock.getElapsedTime();
            line.rotation.x += 0.0005;
            line.rotation.y += 0.001;
            particlesMesh.rotation.y = -0.1 * elapsedTime;
            if (mouse.x !== 0 && mouse.y !== 0) {
                particlesMesh.rotation.x = -mouse.y * (elapsedTime * 0.0008);
                particlesMesh.rotation.y = -mouse.x * (elapsedTime * 0.0008);
            }
            renderer.render(scene, camera);
            frameId = requestAnimationFrame(animate);
        };
        animate();

        return () => {
            cancelAnimationFrame(frameId);
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('resize', handleResize);
            if (currentRef?.contains(renderer.domElement)) {
                currentRef.removeChild(renderer.domElement);
            }
        };
    }, []);

    return <div ref={mountRef} className="fixed top-0 left-0 w-full h-full -z-10" />;
};

export const ThreeJSLandingBackground = React.memo(ThreeJSLandingBackgroundComponent);
