import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .real_ai_agents.openai_critic_agent import OpenAICriticAgent
from .real_ai_agents.openai_patch_agent import OpenAIPatchAgent

logger = logging.getLogger(__name__)


class ProductOrchestrator:
    """
    OLLA2 Product Orchestrator

    Amaç:
      - GitHub PR veya lokal "PR benzeri" context alır
      - Sırasıyla:
          1) Critic agent ile analiz
          2) Patch agent ile unified diff + inline comment üretimi
      - Tek bir birleşik JSON sonucu döner.
    """

    def __init__(self) -> None:
        self.critic_agent = OpenAICriticAgent()
        self.patch_agent = OpenAIPatchAgent()

    async def review_and_patch_pr(self, pr_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Yüksek seviye ürün fonksiyonu:
          - PR metadata + dosyalar + repo bilgisi alır
          - Critic + Patch chain çalıştırır
        """
        sprint_id = pr_context.get("sprint_id") or self._generate_sprint_id()
        pr_title = pr_context.get("title", "Pull Request")
        repo_name = pr_context.get("repo", "unknown/repo")

        goal = f"Review PR '{pr_title}' in repository {repo_name}"

        logger.info(f"[ProductOrchestrator] Starting PR review: {goal}")

        # 1) Critic aşaması
        critic_context = {
            "domain": pr_context.get("domain", "general"),
            "priority": pr_context.get("priority", "medium"),
            "files": pr_context.get("files", []),
            "pr_url": pr_context.get("pr_url"),
            "repo": repo_name,
            "sprint_id": sprint_id,
        }

        critic_result = await self.critic_agent.run(goal, critic_context)
        if critic_result.get("status") != "success":
            logger.error("[ProductOrchestrator] Critic agent failed")
            return self._format_final_result(
                sprint_id=sprint_id,
                goal=goal,
                critic=critic_result,
                patch=None,
            )

        # 2) Patch aşaması – critic output'unu context'e ekle
        patch_context = {
            "files": pr_context.get("files", []),
            "domain": pr_context.get("domain", "general"),
            "priority": pr_context.get("priority", "medium"),
            "previous_output": critic_result,
            "repo": repo_name,
            "sprint_id": sprint_id,
        }

        patch_goal = (
            f"Generate GitHub-native patches for PR '{pr_title}' "
            f"based on critic findings"
        )

        patch_result = await self.patch_agent.run(patch_goal, patch_context)

        final = self._format_final_result(
            sprint_id=sprint_id,
            goal=goal,
            critic=critic_result,
            patch=patch_result,
        )

        logger.info(
            "[ProductOrchestrator] Finished PR review: "
            f"status={final['status']}, "
            f"files_modified={final['summary']['total_files_modified']}"
        )

        return final

    def _format_final_result(
        self,
        sprint_id: str,
        goal: str,
        critic: Dict[str, Any],
        patch: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Critic + Patch sonuçlarını tek bir üst JSON'da toplar.
        """
        status = "success"
        if critic.get("status") != "success":
            status = "failed"
        if patch is not None and patch.get("status") != "success":
            # critic başarılı, patch patlarsa yine de "partial_success" diyebilirsin
            status = "partial_success" if status == "success" else "failed"

        critic_data = critic.get("data") or {}
        patch_data = patch.get("data") if patch else {}

        total_files_modified = patch_data.get("total_files_modified", 0)
        test_required = patch_data.get("test_required", False)
        breaking_risk = patch_data.get("breaking_change_risk", "unknown")

        # Token usage & cost estimate (basit)
        critic_tokens = critic.get("metrics", {}).get("tokens_used", 0)
        patch_tokens = patch.get("metrics", {}).get("tokens_used", 0) if patch else 0
        total_tokens = critic_tokens + patch_tokens
        cost_estimate = self._estimate_cost(total_tokens)

        return {
            "status": status,
            "sprint_id": sprint_id,
            "goal": goal,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files_modified": total_files_modified,
                "test_required": test_required,
                "breaking_change_risk": breaking_risk,
                "security_issues": critic_data.get("security_issues"),
                "quality_issues": critic_data.get("quality_issues"),
                "critic_confidence": critic_data.get("confidence"),
                "patch_confidence": (patch_data.get("confidence") if patch_data else None),
                "cost_estimate_usd": cost_estimate,
                "total_tokens": total_tokens,
            },
            "results": {
                "critic": critic,
                "patch": patch,
            },
        }

    def _generate_sprint_id(self) -> str:
        return datetime.now().strftime("sprint_%Y%m%d_%H%M%S")

    def _estimate_cost(self, total_tokens: int) -> float:
        """
        Çok kabaca bir token→$ mapping (örnek):
          - 1K token ≈ $0.03 (GPT-4 sınıfı)
        Burayı ileride gerçek model fiyatlarıyla güncelleyebilirsin.
        """
        return round((total_tokens / 1000.0) * 0.03, 4)


# Basit manuel test için (örneğin lokal scriptten)
async def _demo():
    orchestrator = ProductOrchestrator()
    dummy_pr = {
        "title": "Fix auth security issues",
        "repo": "example/app",
        "domain": "security",
        "priority": "high",
        "files": [
            {
                "filename": "src/auth.py",
                "content": "def authenticate(...):\n    # insecure demo code\n    ..."
            }
        ],
    }
    result = await orchestrator.review_and_patch_pr(dummy_pr)
    print(result)


if __name__ == "__main__":
    asyncio.run(_demo())
