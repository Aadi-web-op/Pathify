export const playFuturisticSound = (type = 'click') => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (!audioContext) return;
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    gainNode.gain.setValueAtTime(0, audioContext.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.01);

    switch (type) {
        case 'navigate':
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(500, audioContext.currentTime);
            oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 0.1);
            break;
        case 'success':
        case 'levelup':
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(600, audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.2, audioContext.currentTime + 0.05);
            oscillator.frequency.exponentialRampToValueAtTime(1200, audioContext.currentTime + 0.2);
            break;
        case 'click':
        default:
            oscillator.type = 'triangle';
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
            break;
    }

    oscillator.start(audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.00001, audioContext.currentTime + 0.2);
    oscillator.stop(audioContext.currentTime + 0.2);
};