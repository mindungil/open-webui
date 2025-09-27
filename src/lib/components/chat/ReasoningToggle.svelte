<script lang="ts">
  import { get } from "svelte/store";
  import { settings } from "$lib/stores/settings";         // { showReasoning: boolean, ... }
  import { webSearchEnabled } from "$lib/stores/websearch"; // 웹검색 토글 상태 스토어 (프로젝트에 맞춰 경로 조정)

  let showReasoning = false;
  const unsubA = settings.subscribe(v => { showReasoning = !!v.showReasoning; });

  // 웹검색이 켜지면 추론 토글을 강제로 끄고, 버튼을 비활성화
  const unsubB = webSearchEnabled.subscribe((on) => {
    if (on) {
      settings.update(s => ({ ...s, showReasoning: false }));
    }
  });

  // 메모리 정리
  onDestroy(() => { unsubA(); unsubB(); });
</script>

<button
  class="btn"
  on:click={() => settings.update(s => ({ ...s, showReasoning: !get(settings).showReasoning }))}
  disabled={$webSearchEnabled}   <!-- 웹검색 켰을 땐 비활성화 -->
  aria-pressed={showReasoning}
>
  {#if showReasoning} 추론모드: 켜짐 {/if}
  {#if !showReasoning} 추론모드: 꺼짐 {/if}
</button>

