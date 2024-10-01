const banners = document.querySelectorAll('.banner-container img');
let currentBannerIndex = 0;

function changeBanner() {
    banners[currentBannerIndex].classList.remove('active'); 
    currentBannerIndex = (currentBannerIndex + 1) % banners.length; 
    banners[currentBannerIndex].classList.add('active'); 
}

setInterval(changeBanner, 5000);
