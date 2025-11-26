// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize all components
    initThreeJS();
    initScrollReveal();
    initCounters();
    initSmoothScroll();
    initMobileMenu();
    initParallax();
    initNavScrollEffect();
    initSpotlightEffect();
    initPulseOnView();
    
    // Three.js Hero Animation with fallback
    function initThreeJS() {
        const canvas = document.getElementById('heroCanvas');
        if (!canvas) return;
        
        // Check if Three.js is available
        if (typeof THREE === 'undefined') {
            console.warn('Three.js not loaded, using fallback orb');
            createFallbackOrb();
            return;
        }
        
        try {
            // Scene setup
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ 
                canvas: canvas, 
                alpha: true, 
                antialias: true 
            });
            
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            
            // Updated opacity values as requested
            const baseOpacity = 0.2;
            const targetOpacity = 0.5;
            
            // Main orb geometry
            const orbGeometry = new THREE.SphereGeometry(2, 32, 32);
            const orbMaterial = new THREE.MeshPhongMaterial({
                color: 0x3b82f6,
                transparent: true,
                opacity: baseOpacity,
                shininess: 100
            });
            const orb = new THREE.Mesh(orbGeometry, orbMaterial);
            scene.add(orb);
            
            // Inner glow orb with reduced opacity
            const glowGeometry = new THREE.SphereGeometry(2.2, 32, 32);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: 0x60a5fa,
                transparent: true,
                opacity: baseOpacity * 0.5
            });
            const glowOrb = new THREE.Mesh(glowGeometry, glowMaterial);
            scene.add(glowOrb);
            
            // Add a "tablet" 3D object below the orb
            const tabletGeometry = new THREE.BoxGeometry(3.5, 2.2, 0.18);
            const tabletMaterial = new THREE.MeshPhongMaterial({
                color: 0xffffff,
                shininess: 80,
                transparent: true,
                opacity: 0.85
            });
            const tablet = new THREE.Mesh(tabletGeometry, tabletMaterial);
            tablet.position.set(0, -2.5, 0);
            tablet.castShadow = true;
            tablet.receiveShadow = true;
            scene.add(tablet);
            
            // Add a "screen" effect to the tablet
            const screenGeometry = new THREE.PlaneGeometry(3.2, 1.8);
            const screenMaterial = new THREE.MeshBasicMaterial({
                color: 0x3b82f6,
                transparent: true,
                opacity: 0.18
            });
            const screen = new THREE.Mesh(screenGeometry, screenMaterial);
            screen.position.set(0, -2.5, 0.1);
            scene.add(screen);
            
            // Mouse interaction for opacity changes
            let mouseX = 0;
            let mouseY = 0;
            let isMouseOverHero = false;
            let tabletIsHovered = false;
            
            function handleMouseMove(event) {
                const rect = canvas.getBoundingClientRect();
                mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouseY = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                // Check if mouse is over hero section
                const heroSection = document.querySelector('section');
                if (heroSection) {
                    const heroRect = heroSection.getBoundingClientRect();
                    isMouseOverHero = event.clientY >= heroRect.top && event.clientY <= heroRect.bottom;
                }
            }
            
            // Add mouse event listeners to document for better coverage
            document.addEventListener('mousemove', handleMouseMove);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const pointLight1 = new THREE.PointLight(0x3b82f6, 1, 100);
            pointLight1.position.set(10, 10, 10);
            scene.add(pointLight1);
            
            const pointLight2 = new THREE.PointLight(0x8b5cf6, 0.8, 100);
            pointLight2.position.set(-10, -10, 5);
            scene.add(pointLight2);
            
            // Particles with reduced opacity
            const particleCount = 100;
            const particles = new THREE.BufferGeometry();
            const positions = new Float32Array(particleCount * 3);
            const colors = new Float32Array(particleCount * 3);
            
            for (let i = 0; i < particleCount; i++) {
                const i3 = i * 3;
                positions[i3] = (Math.random() - 0.5) * 50;
                positions[i3 + 1] = (Math.random() - 0.5) * 50;
                positions[i3 + 2] = (Math.random() - 0.5) * 30;
                
                colors[i3] = Math.random();
                colors[i3 + 1] = Math.random();
                colors[i3 + 2] = 1;
            }
            
            particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            const particleMaterial = new THREE.PointsMaterial({
                size: 0.1,
                transparent: true,
                opacity: baseOpacity,
                vertexColors: true
            });
            
            const particleSystem = new THREE.Points(particles, particleMaterial);
            scene.add(particleSystem);
            
            // Camera position
            camera.position.z = 10;
            camera.position.y = 1;
            
            // Animation variables
            let time = 0;
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                time += 0.01;
                
                // Rotate main orb
                orb.rotation.x = time * 0.5;
                orb.rotation.y = time * 0.3;
                
                // Rotate glow orb opposite direction
                glowOrb.rotation.x = -time * 0.3;
                glowOrb.rotation.y = -time * 0.5;
                
                // Float orbs
                orb.position.y = Math.sin(time) * 0.5;
                glowOrb.position.y = Math.sin(time * 0.8) * 0.3;
                
                // Animate opacity based on mouse interaction
                if (isMouseOverHero) {
                    orbMaterial.opacity = THREE.MathUtils.lerp(orbMaterial.opacity, targetOpacity, 0.05);
                    glowMaterial.opacity = THREE.MathUtils.lerp(glowMaterial.opacity, targetOpacity * 0.5, 0.05);
                    particleMaterial.opacity = THREE.MathUtils.lerp(particleMaterial.opacity, targetOpacity * 0.8, 0.05);
                } else {
                    orbMaterial.opacity = THREE.MathUtils.lerp(orbMaterial.opacity, baseOpacity, 0.05);
                    glowMaterial.opacity = THREE.MathUtils.lerp(glowMaterial.opacity, baseOpacity * 0.5, 0.05);
                    particleMaterial.opacity = THREE.MathUtils.lerp(particleMaterial.opacity, baseOpacity, 0.05);
                }
                
                // Tablet hover animation
                if (tabletIsHovered) {
                    tablet.position.y = THREE.MathUtils.lerp(tablet.position.y, -1.8, 0.15);
                    tablet.material.opacity = THREE.MathUtils.lerp(tablet.material.opacity, 1, 0.15);
                    screen.material.opacity = THREE.MathUtils.lerp(screen.material.opacity, 0.35, 0.15);
                    tablet.scale.set(1.08, 1.08, 1.08);
                } else {
                    tablet.position.y = THREE.MathUtils.lerp(tablet.position.y, -2.5, 0.15);
                    tablet.material.opacity = THREE.MathUtils.lerp(tablet.material.opacity, 0.85, 0.15);
                    screen.material.opacity = THREE.MathUtils.lerp(screen.material.opacity, 0.18, 0.15);
                    tablet.scale.set(1, 1, 1);
                }
                
                // Animate particles
                const positions = particleSystem.geometry.attributes.position.array;
                for (let i = 0; i < particleCount; i++) {
                    const i3 = i * 3;
                    positions[i3 + 1] += Math.sin(time + i * 0.1) * 0.01;
                }
                particleSystem.geometry.attributes.position.needsUpdate = true;
                
                // Rotate particle system
                particleSystem.rotation.y = time * 0.1;
                
                renderer.render(scene, camera);
            }
            
            animate();
            
            // Handle window resize
            function handleResize() {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }
            
            window.addEventListener('resize', handleResize);
            
        } catch (error) {
            console.error('Three.js initialization failed:', error);
            createFallbackOrb();
        }
    }
    
    // Fallback orb when Three.js is not available
    function createFallbackOrb() {
        const heroSection = document.querySelector('section');
        if (!heroSection) return;
        
        const fallbackOrb = document.createElement('div');
        fallbackOrb.className = 'fallback-orb';
        fallbackOrb.style.opacity = '0.2';
        
        // Add hover effect
        heroSection.addEventListener('mouseenter', () => {
            fallbackOrb.style.opacity = '0.5';
        });
        
        heroSection.addEventListener('mouseleave', () => {
            fallbackOrb.style.opacity = '0.2';
        });
        
        heroSection.appendChild(fallbackOrb);
    }
    
    // Scroll Reveal Animation
    function initScrollReveal() {
        const revealElements = document.querySelectorAll('.reveal');
        
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });
        
        revealElements.forEach(element => {
            revealObserver.observe(element);
        });
    }
    
    // Pulse on view animation for differentiator cards
    function initPulseOnView() {
        const pulseElements = document.querySelectorAll('.pulse-on-view');
        
        const pulseObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('viewed');
                    // Add a slight delay before removing the class to allow for re-triggering
                    setTimeout(() => {
                        entry.target.classList.remove('viewed');
                    }, 1000);
                }
            });
        }, {
            threshold: 0.3,
            rootMargin: '0px 0px -100px 0px'
        });
        
        pulseElements.forEach(element => {
            pulseObserver.observe(element);
        });
    }
    
    // Spotlight effect for cards
    function initSpotlightEffect() {
        const spotlightCards = document.querySelectorAll('.spotlight-card');
        
        spotlightCards.forEach(card => {
            card.addEventListener('mousemove', function(e) {
                const rect = card.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top) / rect.height) * 100;
                
                card.style.setProperty('--mouse-x', `${x}%`);
                card.style.setProperty('--mouse-y', `${y}%`);
            });
            
            card.addEventListener('mouseleave', function() {
                card.style.setProperty('--mouse-x', '50%');
                card.style.setProperty('--mouse-y', '50%');
            });
        });
    }
    
    // Counter Animation - Fixed formatting
    function initCounters() {
        const counters = document.querySelectorAll('[data-count]');
        
        const counterObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const counter = entry.target;
                    const target = parseInt(counter.getAttribute('data-count'));
                    const label = counter.nextElementSibling.textContent.toLowerCase();
                    
                    animateCounter(counter, target, label);
                    counterObserver.unobserve(counter);
                }
            });
        }, { threshold: 0.5 });
        
        counters.forEach(counter => {
            counterObserver.observe(counter);
        });
        
        function animateCounter(element, target, label) {
            let current = 0;
            const increment = target / 50;
            const duration = 2000;
            const stepTime = duration / 50;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                
                // Format numbers based on target and label
                if (target >= 10000) {
                    element.textContent = Math.floor(current / 1000) + 'k+';
                } else if (label.includes('success') || label.includes('rate')) {
                    element.textContent = Math.floor(current) + '%';
                } else if (label.includes('minute') || label.includes('average')) {
                    element.textContent = Math.floor(current) + ' min';
                } else {
                    element.textContent = Math.floor(current) + '%';
                }
            }, stepTime);
        }
    }
    
    // Enhanced Smooth Scroll for Navigation
    function initSmoothScroll() {
        // Handle all navigation links
        const allNavLinks = document.querySelectorAll('a[href^="#"]');
        
        allNavLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    const headerHeight = document.querySelector('nav').offsetHeight;
                    const targetPosition = targetElement.offsetTop - headerHeight - 20;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                    
                    // Close mobile menu if open
                    const mobileMenu = document.getElementById('mobile-menu');
                    if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
                        mobileMenu.classList.add('hidden');
                        mobileMenu.classList.remove('open');
                        
                        // Reset hamburger icon
                        const menuBtn = document.getElementById('mobile-menu-btn');
                        const icon = menuBtn.querySelector('svg');
                        if (icon) {
                            icon.style.transform = 'rotate(0deg)';
                        }
                    }
                }
            });
        });
        
        // Handle button clicks with visual feedback
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            button.addEventListener('click', function(e) {
                // Add visual feedback
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = '';
                }, 150);

                const buttonText = this.textContent.toLowerCase();

                // Handle specific button actions
                if (buttonText.includes('start assessment') || buttonText.includes('get started') || buttonText.includes('watch demo')) {
                    window.location.href = "login.html";
                } else if (buttonText.includes('partner')) {
                    // Simulate partnership inquiry
                    console.log('Partnership inquiry...');
                    alert('Partnership opportunities available! Contact us for more info.');
                }
            });
        });
    }
    
    // Mobile Menu Toggle - Enhanced
    function initMobileMenu() {
        const menuBtn = document.getElementById('mobile-menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');
        
        if (menuBtn && mobileMenu) {
            menuBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                
                const isOpen = !mobileMenu.classList.contains('hidden');
                
                if (isOpen) {
                    // Close menu
                    mobileMenu.classList.add('hidden');
                    mobileMenu.classList.remove('open');
                } else {
                    // Open menu
                    mobileMenu.classList.remove('hidden');
                    mobileMenu.classList.add('open');
                }
                
                // Animate menu button
                const icon = menuBtn.querySelector('svg');
                if (icon) {
                    if (!isOpen) {
                        icon.style.transform = 'rotate(90deg)';
                        icon.style.transition = 'transform 0.3s ease';
                    } else {
                        icon.style.transform = 'rotate(0deg)';
                    }
                }
            });
            
            // Close menu when clicking outside
            document.addEventListener('click', function(e) {
                if (!menuBtn.contains(e.target) && !mobileMenu.contains(e.target)) {
                    mobileMenu.classList.add('hidden');
                    mobileMenu.classList.remove('open');
                    const icon = menuBtn.querySelector('svg');
                    if (icon) {
                        icon.style.transform = 'rotate(0deg)';
                    }
                }
            });

            // Close menu on window resize to desktop
            window.addEventListener('resize', function() {
                if (window.innerWidth >= 768) {
                    mobileMenu.classList.add('hidden');
                    mobileMenu.classList.remove('open');
                    const icon = menuBtn.querySelector('svg');
                    if (icon) {
                        icon.style.transform = 'rotate(0deg)';
                    }
                }
            });
        }
    }
    
    // Parallax Effects
    function initParallax() {
        let ticking = false;
        
        function updateParallax() {
            const scrolled = window.pageYOffset;
            const heroSection = document.querySelector('section');
            
            if (heroSection) {
                const heroHeight = heroSection.offsetHeight;
                const rate = scrolled * -0.5;
                
                // Apply parallax to hero background elements
                const laptop = document.querySelector('.animate-float');
                if (laptop && scrolled < heroHeight) {
                    laptop.style.transform = `translateY(${rate * 0.1}px)`;
                }
                
                // Parallax for Three.js canvas
                const canvas = document.getElementById('heroCanvas');
                if (canvas && scrolled < heroHeight) {
                    canvas.style.transform = `translateY(${rate * 0.05}px)`;
                }
                
                // Parallax for fallback orb
                const fallbackOrb = document.querySelector('.fallback-orb');
                if (fallbackOrb && scrolled < heroHeight) {
                    fallbackOrb.style.transform = `translate(-50%, -50%) translateY(${rate * 0.03}px)`;
                }
                
                // Parallax for floating blobs
                const blob1 = document.querySelector('.blob-1');
                const blob2 = document.querySelector('.blob-2');
                
                if (blob1) {
                    blob1.style.transform = `translateY(${scrolled * 0.1}px)`;
                }
                
                if (blob2) {
                    blob2.style.transform = `translateY(${scrolled * 0.15}px)`;
                }
            }
            
            ticking = false;
        }
        
        function requestParallaxTick() {
            if (!ticking) {
                requestAnimationFrame(updateParallax);
                ticking = true;
            }
        }
        
        window.addEventListener('scroll', requestParallaxTick);
    }
    
    // Navigation Scroll Effect
    function initNavScrollEffect() {
        const nav = document.querySelector('nav');
        let lastScrollY = window.scrollY;
        
        function updateNavOnScroll() {
            const currentScrollY = window.scrollY;
            
            if (currentScrollY > 50) {
                nav.classList.add('nav-scrolled');
            } else {
                nav.classList.remove('nav-scrolled');
            }
            
            // Hide nav on scroll down, show on scroll up
            if (currentScrollY > lastScrollY && currentScrollY > 200) {
                nav.style.transform = 'translateY(-100%)';
                nav.style.transition = 'transform 0.3s ease';
            } else {
                nav.style.transform = 'translateY(0)';
            }
            
            lastScrollY = currentScrollY;
        }
        
        let scrollTicking = false;
        window.addEventListener('scroll', function() {
            if (!scrollTicking) {
                requestAnimationFrame(updateNavOnScroll);
                scrollTicking = true;
                setTimeout(() => scrollTicking = false, 10);
            }
        });
    }
    
    // Button Click Animations
    function initButtonAnimations() {
        const buttons = document.querySelectorAll('button, .btn');
        
        buttons.forEach(button => {
            button.addEventListener('click', function(e) {
                // Create ripple effect
                const ripple = document.createElement('span');
                const rect = button.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.6);
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    animation: ripple 0.6s linear;
                    pointer-events: none;
                `;
                
                button.style.position = 'relative';
                button.style.overflow = 'hidden';
                button.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
        
        // Add ripple animation CSS
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Initialize button animations
    initButtonAnimations();
    
    // Initialize additional visual effects
    function initVisualEffects() {
        // Add random delays to floating animations for more natural movement
        const blobs = document.querySelectorAll('.floating-blob');
        blobs.forEach((blob, index) => {
            blob.style.animationDelay = `${index * 5}s`;
        });
        
        // Add intersection observer for step icons to trigger glow effect
        const stepIcons = document.querySelectorAll('.step-icon-ring');
        const stepObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.style.boxShadow = '0 0 20px rgba(59, 130, 246, 0.4)';
                        entry.target.style.transform = 'scale(1.05)';
                        
                        setTimeout(() => {
                            entry.target.style.boxShadow = '';
                            entry.target.style.transform = '';
                        }, 1000);
                    }, Math.random() * 500); // Random delay for staggered effect
                }
            });
        }, {
            threshold: 0.5
        });
        
        stepIcons.forEach(icon => {
            stepObserver.observe(icon);
        });
    }
    
    // Initialize visual effects
    initVisualEffects();
    
    // Error Handling
    window.addEventListener('error', function(e) {
        console.error('JavaScript Error:', e.error);
    });
    
    // Initialize accessibility features
    function initAccessibility() {
        // Add keyboard navigation support
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });
        
        document.addEventListener('mousedown', function() {
            document.body.classList.remove('keyboard-navigation');
        });
        
        // Add focus styles for keyboard navigation
        const style = document.createElement('style');
        style.textContent = `
            .keyboard-navigation *:focus {
                outline: 2px solid #3b82f6 !important;
                outline-offset: 2px !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Initialize accessibility features
    initAccessibility();
    
    console.log('🚀 Pathify application with enhanced hero text clarity and improved functionality initialized successfully!');
});